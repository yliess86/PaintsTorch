import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision.models as M

VGG16_PATH = './res/model/vgg16-397923af.pth'
I2V_PATH   = './res/model/i2v.pth'

class ResNetXBottleneck(nn.Module):
    def __init__(self, in_channels=256, out_channels=256, stride=1, cardinality=32, dilate=1):
        super(ResNetXBottleneck, self).__init__()
        D                 = out_channels // 2
        self.out_channels = out_channels
        self.conv_reduce  = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_conv    = nn.Conv2d(D, D, kernel_size=2 + stride, stride=stride, padding=dilate, dilation=dilate, groups=cardinality, bias=False)
        self.conv_expand  = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.shortcut     = nn.Sequential()

        if stride != 1: self.shortcut.add_module('shortcut', nn.AvgPool2d(2, stride=2))

    def forward(self, X):
        bottleneck = F.leaky_relu(self.conv_reduce(X), 0.2, True)
        bottleneck = F.leaky_relu(self.conv_conv(bottleneck), 0.2, True)
        bottleneck = self.conv_expand(bottleneck)
        X          = self.shortcut(X)

        return X + bottleneck

class Generator(nn.Module):
    def __init__(self, ngf=64, in_channels=1, out_channels=3):
        super(Generator, self).__init__()
        self.toH     = nn.Sequential(
            nn.Conv2d(4, ngf, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(0.2, True)
        )
        self.to0     = nn.Sequential(nn.Conv2d(in_channels, ngf // 2, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(0.2, True))
        self.to1     = nn.Sequential(nn.Conv2d(ngf // 2, ngf, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True))
        self.to2     = nn.Sequential(nn.Conv2d(ngf, ngf * 2, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True))
        self.to3     = nn.Sequential(nn.Conv2d(ngf * 3, ngf * 4, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True))
        self.to4     = nn.Sequential(nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True))

        tunnel4      = nn.Sequential(*[ResNetXBottleneck(ngf * 8, ngf * 8, cardinality=32, dilate=1) for _ in range(20)])
        self.tunnel4 = nn.Sequential(
            nn.Conv2d(ngf * 8 + 512, ngf * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            tunnel4,
            nn.Conv2d(ngf * 8, ngf * 4 * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )

        depth        = 2
        tunnel       = [ResNetXBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1) for _ in range(depth)] + \
                       [ResNetXBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2) for _ in range(depth)] + \
                       [ResNetXBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=4) for _ in range(depth)] + \
                       [ResNetXBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=2),
                        ResNetXBottleneck(ngf * 4, ngf * 4, cardinality=32, dilate=1)]
        tunnel3      = nn.Sequential(*tunnel)
        self.tunnel3 = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            tunnel3,
            nn.Conv2d(ngf * 4, ngf * 2 * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )

        tunnel       = [ResNetXBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1) for _ in range(depth)] + \
                       [ResNetXBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2) for _ in range(depth)] + \
                       [ResNetXBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=4) for _ in range(depth)] + \
                       [ResNetXBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=2),
                        ResNetXBottleneck(ngf * 2, ngf * 2, cardinality=32, dilate=1)]
        tunnel2      = nn.Sequential(*tunnel)
        self.tunnel2 = nn.Sequential(
            nn.Conv2d(ngf * 4, ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            tunnel2,
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )

        tunnel       = [ResNetXBottleneck(ngf, ngf, cardinality=16, dilate=1)] + \
                       [ResNetXBottleneck(ngf, ngf, cardinality=16, dilate=2)] + \
                       [ResNetXBottleneck(ngf, ngf, cardinality=16, dilate=4)] + \
                       [ResNetXBottleneck(ngf, ngf, cardinality=16, dilate=2),
                        ResNetXBottleneck(ngf, ngf, cardinality=16, dilate=1)]
        tunnel1      = nn.Sequential(*tunnel)
        self.tunnel1 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            tunnel1,
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, True)
        )

        self.exit    = nn.Conv2d(ngf, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, sketch, hint, sketch_feat):
        hint = self.toH(hint)
        x0   = self.to0(sketch)
        x1   = self.to1(x0)
        x2   = self.to2(x1)
        x3   = self.to3(torch.cat([x2, hint], 1))
        x4   = self.to4(x3)

        x    = self.tunnel4(torch.cat([x4, sketch_feat], 1))
        x    = self.tunnel3(torch.cat([x, x3], 1))
        x    = self.tunnel2(torch.cat([x, x2], 1))
        x    = self.tunnel1(torch.cat([x, x1], 1))
        x    = torch.tanh(self.exit(torch.cat([x, x0], 1)))

        return x

class Discriminator(nn.Module):
    def __init__(self, ndf=64):
        super(Discriminator, self).__init__()

        self.feed  = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=7, stride=1, padding=3, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),

            ResNetXBottleneck(ndf, ndf, cardinality=8, dilate=1),
            ResNetXBottleneck(ndf, ndf, cardinality=8, dilate=1, stride=2),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),

            ResNetXBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1),
            ResNetXBottleneck(ndf * 2, ndf * 2, cardinality=8, dilate=1, stride=2),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, True),

            ResNetXBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1),
            ResNetXBottleneck(ndf * 4, ndf * 4, cardinality=8, dilate=1, stride=2),
        )

        self.feed2 = nn.Sequential(
            nn.Conv2d(ndf * 12, ndf * 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            ResNetXBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNetXBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),
            ResNetXBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNetXBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),
            ResNetXBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            ResNetXBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1, stride=2),
            ResNetXBottleneck(ndf * 8, ndf * 8, cardinality=8, dilate=1),
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.2, True)
        )

        self.out = nn.Linear(512, 1)

    def forward(self, color, sketch_feat):
        x   = self.feed(color)
        x   = self.feed2(torch.cat([x, sketch_feat], 1))
        out = self.out(x.view(color.size(0), -1))

        return out

class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()
        vgg          = M.vgg16()
        vgg.load_state_dict(torch.load(VGG16_PATH))
        vgg.features = nn.Sequential(*list(vgg.features.children())[:9])
        self.model   = vgg.features

        self.register_buffer('mean', torch.FloatTensor([0.485 - 0.5, 0.456 - 0.5, 0.406 - 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, images):
        return self.model((images.mul(0.5) - self.mean) / self.std)

class Illustration2Vec(nn.Module):
    def __init__(self, path=I2V_PATH):
        super(Illustration2Vec, self).__init__()
        i2v_model  = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(512, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(1024, 1024, (3, 3), (1, 1), (1, 1)),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(1024, 1539, (3, 3), (1, 1), (1, 1)),
            nn.AvgPool2d((7, 7), (1, 1), (0, 0), ceil_mode=True)
        )
        i2v_model.load_state_dict(torch.load(path))
        i2v_model  = nn.Sequential(*list(i2v_model.children())[:15])
        self.model = i2v_model

        self.register_buffer('mean', torch.FloatTensor([164.76139251, 167.47864617, 181.13838569]).view(1, 3, 1, 1))

    def forward(self, images):
        images = F.avg_pool2d(images, 2, 2)
        images = images.mul(0.5).add(0.5).mul(255)

        return self.model(images.expand(-1, 3, 256, 256) - self.mean)
