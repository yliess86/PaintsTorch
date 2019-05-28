import numpy as np

from skimage.color import gray2rgb
from PIL import Image

def tensor2img(X, fun):
    (b, c, h, w) = X.size()
    X            = X[0].permute(1, 2, 0).detach().cpu().numpy()
    if c == 1: X = gray2rgb(X.reshape((w, h)))
    X            = fun(X).astype(np.uint8)

    return X

def mosaic(L, size):
    L = [np.array(Image.fromarray(l).resize(size)) for l in L]
    L = np.hstack(L)
    L = Image.fromarray(L)

    return L

def tensors2vizu(size, dest, strokes=None, col=None, fcol=None, sketch=None, fsketch=None):
    L = []

    if sketch is not None:
        sketch = tensor2img(  sketch, lambda x: (x + 1.0) * 0.5 * 255.0)
        L.append(sketch)
    if col is not None:
        col = tensor2img(        col, lambda x: (x + 1.0) * 0.5 * 255.0)
        L.append(col)

    if strokes is not None:
        strokes = tensor2img(strokes, lambda x: (x + 1.0) * 0.5 * 255.0)
        L.append(strokes)

    if fcol is not None:
        fcol = tensor2img(      fcol, lambda x: (x + 1.0) * 0.5 * 255.0)
        L.append(fcol)
    if fsketch is not None:
        fsketch = tensor2img(fsketch, lambda x: (x + 1.0) * 0.5 * 255.0)
        L.append(fsketch)

    if len(L) > 0:
        img = mosaic(L, size)
        img.save(dest)
