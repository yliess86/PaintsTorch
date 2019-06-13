import torch.nn as nn
import numpy as np
import warnings
import torch
import os

from model import Generator, Illustration2Vec
from data import CreateValidLoader
from eval import Evaluate

from collections import namedtuple

def evaluate(seed, constant, experience_dir, valid_path):
    if not os.path.exists(experience_dir):
        os.mkdir(experience_dir)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    generator     = nn.DataParallel(Generator(constant, in_channels=1, out_channels=3)).cuda()
    i2v           = nn.DataParallel(Illustration2Vec()).cuda()

    if os.path.isfile(os.path.join(experience_dir, 'model/generator.pth')):
        checkG = torch.load(os.path.join(experience_dir, 'model/generator.pth'))
        generator.load_state_dict(checkG['generator'])
    else:
        print('[ERROR] Need generator checkpoint!')
        exit(1)

    valloader     = CreateValidLoader(valid_path, batch_size, img_size)
    config        = {
        'target_npz' : './res/model/fid_stats_color.npz',
        'corps'      : 2,
        'image_size' : img_size[0]
    }
    evaluate      = Evaluate(generator, i2v, valloader, namedtuple('Config', config.keys())(*config.values()))

    generator.eval()
    discriminator.eval()

    fid, fid_var = evaluate()
    print(f'\n===================\nFID = {fid} +- {fid_var}\n===================\n')

    with open(os.path.join(experience_dir, 'evaluate_fid.csv'), 'w') as f:
        f.write(f'evaluate;{fid};{fid_var}\n')
