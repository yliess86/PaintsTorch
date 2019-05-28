import scipy.interpolate as interpolate
import numpy as np
import torch

from torch.autograd import Variable
from PIL import Image, ImageDraw

class Stroke(object):
    def __init__(self, thickness, max_points, move_range):
        assert isinstance(thickness, (tuple, int))
        assert isinstance(max_points, int)
        assert max_points > 1
        assert isinstance(move_range, (tuple, int))

        self.thickness  = thickness
        self.max_points = max_points
        self.move_range = move_range

        self.B          = (  0,   0,   0)
        self.G          = (128, 128, 128)
        self.W          = (255, 255, 255)

    def __call__(self, mask, strokes, img):
        MASK, STROKES = None, None

        while MASK is None or STROKES is None:
            try:
                w, h = mask.width, mask.height
                n    = np.random.randint(1, self.max_points)
                t    = np.random.randint(*self.thickness)
                assert t * 2 < w and t * 2 < h

                draw_mask    = ImageDraw.Draw(mask)
                draw_strokes = ImageDraw.Draw(strokes)

                pos   = [np.random.randint(0, w), np.random.randint(0, h)]
                color = tuple(img[pos[0], pos[1]])
                move  = [
                    np.random.randint(1, self.move_range[0]),
                    np.random.randint(1, self.move_range[1])
                ]

                x = np.random.randint(pos[0] - move[0], pos[0] + move[0], n)
                y = np.random.randint(pos[1] - move[1], pos[1] + move[1], n)

                X = np.linspace(x.min(), x.max(), 1000) if n > 4 else x
                Y = interpolate.spline(x, y, X, order=4) if n > 4 else y

                pts  = [(X[i], Y[i]) for i in range(X.shape[-1])]
                offs = t * 0.5 - 1

                draw_mask.line(pts, width=t, fill=self.W, joint='curve')
                draw_strokes.line(pts, width=t, fill=color, joint='curve')
                for i in range(X.shape[-1]):
                    a = X[i] - offs, Y[i] - offs
                    b = X[i] + offs, Y[i] + offs
                    draw_mask.ellipse((*a, *b), fill=self.W)
                    draw_strokes.ellipse((*a, *b), fill=color)

                MASK    = mask
                STROKES = strokes

            except:
                continue

        return MASK, STROKES

class Hint(object):
    def __init__(self, size, n, thickness, max_points, move_range):
        assert isinstance(size, (tuple, int))
        assert isinstance(n, int)

        self.size   = size
        self.n      = n
        self.stroke = Stroke(thickness, max_points, move_range)

    def __call__(self, X, htranform):
        X       = np.array(Image.fromarray(X.astype(np.uint8)).resize(self.size, Image.BICUBIC))
        n       = np.random.randint(0, self.n)
        MASK    = Image.new('RGB', self.size, self.stroke.B)
        STROKES = Image.new('RGB', self.size, self.stroke.G)

        if n != 0:
            for i in range(n):
                MASK, STROKES = self.stroke(MASK, STROKES, X)

        w, h    = MASK.width, MASK.height

        MASK    = MASK.resize((int(w * 0.8), int(h * 0.8)), Image.ANTIALIAS)
        MASK    = MASK.resize((w, h), Image.ANTIALIAS)
        MASK    = MASK.convert('L')
        MASK    = Variable(torch.from_numpy(np.array(MASK)), requires_grad=False).unsqueeze(2)
        MASK    = MASK / 255.0
        MASK    = MASK.permute(2, 0, 1).float()

        STROKES = STROKES.resize((int(w * 0.8), int(h * 0.8)), Image.ANTIALIAS)
        STROKES = STROKES.resize((w, h), Image.ANTIALIAS)
        STROKES = Variable(torch.from_numpy(np.array(STROKES)), requires_grad=False)
        STROKES = htranform(STROKES.permute(2, 0, 1).float() / 255.)

        HINT    = torch.cat((STROKES, MASK), 0)

        return HINT
