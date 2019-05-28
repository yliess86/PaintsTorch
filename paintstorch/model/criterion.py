import torch.nn as nn
import torch

from torch.autograd import grad

class GradPenalty(nn.Module):
    def __init__(self, gpW):
        super(GradPenalty, self).__init__()
        self.gpW = gpW

    def forward(self, discriminator, real, fake, sketch_feat):
        alpha = torch.rand(real.size(0), 1, 1, 1)
        if real.is_cuda: alpha = alpha.cuda()

        interp               = alpha * real + ((1 - alpha) * fake)
        interp.requires_grad = True
        disc_interp          = discriminator(interp, sketch_feat)
        grads                = grad(
            outputs      = disc_interp,
            inputs       = interp,
            grad_outputs = torch.ones(disc_interp.size()).cuda() \
                           if real.is_cuda\
                           else torch.ones(disc_interp.size()),
            create_graph = True,
            retain_graph = True,
            only_inputs  = True
        )[0]

        return ((grads.norm(2, dim=1) - 1) ** 2).mean() * self.gpW
