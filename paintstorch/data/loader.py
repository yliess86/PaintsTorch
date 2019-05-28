import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.stats as stats
import numpy as np
import torch
import os

from data import xDoG
from PIL import Image

from torchvision.transforms import Resize, CenterCrop
from torch.utils.data.sampler import Sampler

EXT = ['.jpeg', '.JPEG', '.jpg', '.JPG', '.png', '.PNG', '.ppm', '.PPM', '.bpm', '.BPM']

class RandomCrop(object):
    def __init__(self, size):
        assert isinstance(size, (tuple, int))
        self.size = size

    def __call__(self, colored, sketch):
        w, h    = colored.width, colored.height
        scale   = w / self.size[0] if h > w else h / self.size[1]

        w, h    = int(w / scale), int(h / scale)
        colored = colored.resize((w, h), Image.BICUBIC)
        sketch  = sketch.resize((w, h), Image.BICUBIC)

        x       = np.random.randint(0, w - self.size[0]) if w != self.size[0] else 0
        y       = np.random.randint(0, h - self.size[1]) if h != self.size[1] else 0

        colored = colored.crop((x, y, x + self.size[0], y + self.size[1]))
        sketch  = sketch.crop((x, y, x + self.size[0], y + self.size[1]))

        return colored, sketch

class ImageFolder(data.Dataset):
    def __init__(self, root, img_size, method='strokes', mask_gen=None, transform=None, stransform=None, htransform=None):
        self.root       = root
        self.mask_gen   = mask_gen
        self.transform  = transform
        self.stransform = stransform
        self.htransform = htransform
        self.files      = self.init(root)
        self.crop       = RandomCrop(img_size)
        self.method     = method
        self.img_size   = img_size

    def init(self, dir):
        imgs = [
            os.path.join(root, file)
            for root, dirs, files in os.walk(dir)
            for file in files
        ]
        return [img for img in imgs if self.is_image(img)]

    def is_image(self, filename):
        return any(filename.endswith(extension) for extension in EXT)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        colored = self.load_image(self.files[idx])
        xdog    = xDoG(0.95, 1e9, -1e-1, 4.5, np.random.choice([0.3, 0.4, 0.5]))
        sketch  = xdog(np.array(colored) / 255.0)
        sketch  = Image.fromarray((sketch * 255.0).astype(np.uint8)).convert('RGB')

        colored, sketch = self.crop(colored, sketch)
        if np.random.random() < 0.5:
            colored = colored.transpose(Image.FLIP_LEFT_RIGHT)
            sketch  = sketch.transpose(Image.FLIP_LEFT_RIGHT)

        coin = np.random.random()
        hint = None

        if coin >= 0.5 and self.method == 'strokes':
            hint = self.mask_gen(
                transforms.ToTensor()(colored).permute(2, 1, 0).detach().cpu().numpy() * 255.,
                self.htransform
            )

        if coin >= 0.5 and self.method == 'random':
            mu, sigma = 1, 5e-3
            X         = stats.truncnorm((0 - mu) / sigma, (1 - mu) / sigma, loc=mu, scale=sigma)
            maskS     = self.img_size[0] // 4
            mask      = torch.rand(1, maskS, maskS).ge(X.rvs(1)[0]).float()
            vcolored  = self.transform(colored.resize((self.img_size[0] // 4, self.img_size[1] // 4)))
            hint      = torch.cat((vcolored * mask, mask), 0)

        colored  = self.transform(colored)
        sketch   = self.stransform(sketch)


        if hint is None:
            hint = torch.cat((
                self.htransform(torch.ones((3, colored.size(1) // 4, colored.size(2) // 4)) * 0.5),
                torch.zeros((1, colored.size(1) // 4, colored.size(2) // 4))
            ), 0)

        return colored, sketch, hint

    def load_image(self, path):
        return Image.open(path).convert('RGB')

def CreateTrainLoader(root, batch_size, mask_gen, img_size, method):
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def jitter(x):
        ran = np.random.uniform(0.7, 1)
        return x * ran + 1 - ran

    stransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(jitter),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        lambda x: x.mean(0).unsqueeze(0)
    ])

    htransform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    dataset    = ImageFolder(root, img_size, method, mask_gen, transform, stransform, htransform)
    loader     = data.DataLoader(
        dataset,
        batch_size,
        shuffle     = True,
        pin_memory  = True,
        num_workers = 8
    )

    return loader

def CreateValidLoader(root, batch_size, img_size):
    transform  = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    stransform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        lambda x: x.mean(0).unsqueeze(0)
    ])

    htransform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    dataset    = ImageFolder(root, img_size, None, None, transform, stransform, htransform)
    loader     = data.DataLoader(
        dataset,
        batch_size,
        shuffle     = False,
        pin_memory  = True,
        num_workers = 8
    )

    return loader
