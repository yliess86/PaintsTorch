import numpy as np

from eval import FID

class Evaluate(object):
    def __init__(self, G, I, dataloader, config, dims=2048):
        self.fid        = FID()
        self.G          = G
        self.I          = I
        self.dataloader = dataloader
        self.config     = config
        self.dims       = dims

    def __call__(self):
        fids      = []
        fid_value = 0

        for _ in range(3):
            fid        = self.fid(
                self.G,
                self.I,
                self.dataloader,
                self.config,
                self.dims
            )
            fid_value += fid
            fids.append(fid)

        fid_value /= 3

        return fid_value, np.var(fids)
