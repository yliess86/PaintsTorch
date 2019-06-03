import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
import warnings
import torch
import os

from model import Generator, Discriminator, Features, Illustration2Vec
from data import CreateTrainLoader, CreateValidLoader
from model import StepLRScheduler
from model import GradPenalty
from data import tensors2vizu
from eval import Evaluate
from data import Hint

from collections import namedtuple
from skimage.color import gray2rgb
from tqdm import tqdm
from PIL import Image

def train(double, seed, constant, experience_dir, train_path, valid_path, batch_size, epochs, drift, adwd, method):
    if not os.path.exists(experience_dir):
        os.mkdir(experience_dir)
    if not os.path.exists(os.path.join(experience_dir, 'model')):
        os.mkdir(os.path.join(experience_dir, 'model'))
    if not os.path.exists(os.path.join(experience_dir, 'vizu')):
        os.mkdir(os.path.join(experience_dir, 'vizu'))

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    generator     = nn.DataParallel(Generator(constant, in_channels=1, out_channels=3)).cuda()
    discriminator = nn.DataParallel(Discriminator(constant)).cuda()
    features      = nn.DataParallel(Features()).cuda()
    i2v           = nn.DataParallel(Illustration2Vec()).cuda()
    sketcher      = nn.DataParallel(Generator(constant, in_channels=3, out_channels=1)).cuda() if double else None

    optimizerG    = optim.Adam(
        list(generator.parameters()) + list(sketcher.parameters()) \
        if double \
        else generator.parameters(),
        lr=1e-4, betas=(0.5, 0.9)
    )
    optimizerD    = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.9))

    lr_schedulerG = StepLRScheduler(optimizerG, [125000], [0.1], 1e-4, 0, 0)
    lr_schedulerD = StepLRScheduler(optimizerD, [125000], [0.1], 1e-4, 0, 0)

    if os.path.isfile(os.path.join(experience_dir, 'model/generator.pth')):
        checkG = torch.load(os.path.join(experience_dir, 'model/generator.pth'))
        generator.load_state_dict(checkG['generator'])
        if double: sketcher.load_state_dict(checkG['sketcher'])
        optimizerG.load_state_dict(checkG['optimizer'])

    if os.path.isfile(os.path.join(experience_dir, 'model/discriminator.pth')):
        checkD = torch.load(os.path.join(experience_dir, 'model/discriminator.pth'))
        discriminator.load_state_dict(checkD['discriminator'])
        optimizerD.load_state_dict(checkD['optimizer'])

    for param in features.parameters(): param.requires_grad = False
    mse           = nn.MSELoss().cuda()
    grad_penalty  = GradPenalty(10).cuda()

    img_size      = (512, 512)
    mask_gen      = Hint((img_size[0] // 4, img_size[1] // 4), 120, (1, 4), 5, (10, 10))
    dataloader    = CreateTrainLoader(train_path, batch_size, mask_gen, img_size, method=method)
    iterator      = iter(dataloader)

    valloader     = CreateValidLoader(valid_path, batch_size, img_size)
    config        = {
        'target_npz' : './res/model/fid_stats_color.npz',
        'corps'      : 2,
        'image_size' : img_size[0]
    }
    evaluate      = Evaluate(generator, i2v, valloader, namedtuple('Config', config.keys())(*config.values()))

    start_epoch   = 0
    if os.path.isfile(os.path.join(experience_dir, 'model/generator.pth')):
        checkG      = torch.load(os.path.join(experience_dir, 'model/generator.pth'))
        start_epoch = checkG['epoch'] + 1

    for epoch in range(start_epoch, epochs):
        batch_id = -1

        pbar = tqdm(total=len(dataloader), desc=f'Epoch [{(epoch + 1):06d}/{epochs}]')
        id   = 0
        dgp  = 0.
        gc   = 0.
        s    = 0. if double else None

        while batch_id < len(dataloader):
            lr_schedulerG.step(epoch)
            lr_schedulerD.step(epoch)

            generator.train()
            discriminator.train()
            if double: sketcher.train()

            # =============
            # DISCRIMINATOR
            # =============
            for p in discriminator.parameters(): p.requires_grad = True
            for p in generator.parameters(): p.requires_grad = False
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            batch_id += 1
            try:
                colored, sketch, hint = iterator.next()
            except StopIteration:
                iterator                       = iter(dataloader)
                colored, sketch, hint = iterator.next()

            real_colored  = colored.cuda()
            real_sketch   = sketch.cuda()
            hint          = hint.cuda()

            with torch.no_grad():
                feat_sketch  = i2v(real_sketch).detach()
                fake_colored = generator(real_sketch, hint, feat_sketch).detach()

            errD_fake = discriminator(fake_colored, feat_sketch).mean(0).view(1)
            errD_fake.backward(retain_graph=True)

            errD_real = discriminator(real_colored, feat_sketch).mean(0).view(1)
            errD      = errD_real - errD_fake

            errD_realer = -1 * errD_real + errD_real.pow(2) * drift
            errD_realer.backward(retain_graph=True)

            gp = grad_penalty(discriminator, real_colored, fake_colored, feat_sketch)
            gp.backward()

            optimizerD.step()
            pbar.update(1)

            dgp += errD_realer.item() + gp.item()

            # =============
            # GENERATOR
            # =============
            for p in generator.parameters(): p.requires_grad = True
            for p in discriminator.parameters(): p.requires_grad = False
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            batch_id                += 1
            try:
                colored, sketch, hint = iterator.next()
            except StopIteration:
                iterator                       = iter(dataloader)
                colored, sketch, hint = iterator.next()

            real_colored  = colored.cuda()
            real_sketch   = sketch.cuda()
            hint          = hint.cuda()

            with torch.no_grad():
                feat_sketch = i2v(real_sketch).detach()

            fake_colored = generator(real_sketch, hint, feat_sketch)

            errD = discriminator(fake_colored, feat_sketch)
            errG = -1 * errD.mean() * adwd
            errG.backward(retain_graph=True)

            feat1 = features(fake_colored)
            with torch.no_grad():
                feat2 = features(real_colored)

            contentLoss = mse(feat1, feat2)
            contentLoss.backward()

            optimizerG.step()
            pbar.update(1)

            gc += errG.item() + contentLoss.item()

            # =============
            # SKETCHER
            # =============
            if double:
                for p in generator.parameters(): p.requires_grad = True
                for p in discriminator.parameters(): p.requires_grad = False
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                batch_id                += 1
                try:
                    colored, sketch, hint = iterator.next()
                except StopIteration:
                    iterator                       = iter(dataloader)
                    colored, sketch, hint = iterator.next()

                real_colored  = colored.cuda()
                real_sketch   = sketch.cuda()
                hint          = hint.cuda()

                with torch.no_grad():
                    feat_sketch = i2v(real_sketch).detach()

                fake_colored = generator(real_sketch, hint, feat_sketch)
                fake_sketch  = sketcher(fake_colored, hint, feat_sketch)
                errS         = mse(fake_sketch, real_sketch)
                errS.backward()

                optimizerG.step()
                pbar.update(1)

                s += errS.item()

            pbar.set_postfix(**{
                'dgp': dgp / (id + 1),
                'gc' : gc / (id + 1),
                's'  : s / (id + 1) if double else None
            })

            # =============
            # PLOT
            # =============
            generator.eval()
            discriminator.eval()

            if id % 20 == 0:
                tensors2vizu(img_size, os.path.join(experience_dir, f'vizu/out_{epoch}_{id}_{batch_id}.jpg'),
                    **{
                        'strokes': hint[:, :3, ...],
                        'col'    : real_colored,
                        'fcol'   : fake_colored,
                        'sketch' : real_sketch,
                        'fsketch': fake_sketch if double else None
                    }
                )

            id += 1

        pbar.close()

        torch.save({
            'generator': generator.state_dict(),
            'sketcher' : sketcher.state_dict() if double else None,
            'optimizer': optimizerG.state_dict(),
            'double'   : double,
            'epoch'    : epoch
        }, os.path.join(experience_dir, 'model/generator.pth'))

        torch.save({
            'discriminator': discriminator.state_dict(),
            'optimizer'    : optimizerD.state_dict(),
            'epoch'        : epoch
        }, os.path.join(experience_dir, 'model/discriminator.pth'))

        if epoch % 20 == 0:
            fid, fid_var = evaluate()
            print(f'\n===================\nFID = {fid} +- {fid_var}\n===================\n')

            with open(os.path.join(experience_dir, 'fid.csv'), 'a+') as f:
                f.write(f'{epoch};{fid};{fid_var}\n')
