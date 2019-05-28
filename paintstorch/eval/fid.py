import numpy as np
import torch
import os

from torch.nn.functional import adaptive_avg_pool2d
from eval import InceptionV3
from scipy import linalg
from tqdm import tqdm

class FID(object):
    def __call__(self, G, I, dataloader, config, dims):
        assert os.path.exists(config.target_npz), \
               f'Invalid path {config.target_npz}'

        block_idx = InceptionV3.BLOCK_INDEX_BY_DIMS[dims]
        model     = torch.nn.DataParallel(InceptionV3([block_idx]))
        model     = model.cuda()
        model.eval()

        f         = np.load(config.target_npz)
        mu1, sig1 = f['mu'][:], f['sigma'][:]
        f.close()

        acts      = []
        print()
        for _ in tqdm(range(config.corps), desc='Computing FID'):
            iterator = iter(dataloader)

            for _, real_sketch, _ in tqdm(iterator):
                real_sketch = real_sketch.cuda()
                zhint       = torch.zeros(
                    real_sketch.shape[0],
                    4,
                    config.image_size // 4,
                    config.image_size // 4
                ).float().cuda()

                with torch.no_grad():
                    feat_sketch = I(real_sketch)
                    fake_color  = G(real_sketch, zhint, feat_sketch).squeeze()
                    pred        = model(adaptive_avg_pool2d(
                        fake_color.mul(0.5).add(0.5),
                        output_size=299
                    ))[0]

                if pred.shape[2] != 1 or pred.shape[3] != 1:
                    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

                acts.append(pred.cpu().data.numpy().squeeze())

        act  = np.concatenate(acts, 0)
        mu2  = np.mean(act, axis=0)
        sig2 = np.cov(act, rowvar=False)
        fid  = self._frechet_dst(mu1, sig1, mu2, sig2)
        print()

        return fid

    def _frechet_dst(self, mu1, sig1, mu2, sig2, eps=1e-6):
        mu1  = np.atleast_1d(mu1)
        sig1 = np.atleast_1d(sig1)
        mu2  = np.atleast_1d(mu2)
        sig2 = np.atleast_1d(sig2)

        assert mu1.shape == mu2.shape, 'Train & Test not same lengths'
        assert sig1.shape == sig2.shape, 'Train & Test not same dimensions'

        diff    = mu1 - mu2
        covmean = linalg.sqrtm(sig1.dot(sig2), disp=False)[0]

        if not np.isfinite(covmean).all():
            print(
                'FID produces singular product.',
                f'Adding {eps} to diagonal of cov estimates.'
            )
            offset  = np.eye(sig1.shape[0]) * eps
            covmean = linalg.sqrtm((sig1 + offset).dot(sig2 + offset))

        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f'Imaginary Component {m}')
            covmean = covmean.real

        tr  = np.trace(covmean)
        res = diff.dot(diff) + np.trace(sig1) + np.trace(sig2) - 2 * tr

        return res
