import numbers
import collections
import torch
from math import sqrt
import torch.nn as nn
import numpy as np


class PriorBox(nn.Module):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg, order='khw'):
        super(PriorBox, self).__init__()
        self.min_sizes = cfg.get('min_sizes', None)
        self.max_sizes = cfg.get('max_sizes', None)
        self.scales = cfg.get('scales', None)
        self.aspect_ratios = cfg.get('aspect_ratios', None)
        self.sizes = cfg.get('sizes', None)
        self.clip = cfg.get('clip', False)
        self.rotations = cfg.get('rotations', None)
        if order == 'khw':
            self.khw_order = True
        elif order == 'hwk':
            self.khw_order = False
        else:
            raise NotImplementedError(order)

        if self.min_sizes is None and self.sizes is None:
            raise RuntimeError("invalid config: both min_sizes and sizes are None")
        if self.min_sizes and self.sizes:
            raise RuntimeError("invalid config: both min_sizes and sizes are not None")

        if self.sizes:
            self.n_layer = len(self.sizes)
        elif self.min_sizes:
            self.n_layer = len(self.min_sizes)

        for sizes in (self.min_sizes, self.max_sizes, self.scales, self.sizes, self.aspect_ratios):
            if sizes is not None:
                assert self.n_layer == len(sizes)

        # cache to reduce computation
        self.image_size = None
        self.priors_size = None
        self.priors = None

    def forward(self, input):
        image_size, sources = input

        priors_size = []
        for x in sources:
            priors_size.append((x.size(2), x.size(3)))
        priors_size = np.asarray(priors_size)
        image_size = np.asarray(image_size)
        if (self.image_size is not None and np.all(self.image_size == image_size) and
                self.priors is not None and np.all(self.priors_size == priors_size)):
            self.priors = self.priors.to(sources[0])
            return self.priors

        if self.sizes:
            mean = self.compute_priors_with_size(image_size, priors_size)
        elif self.scales:
            mean = self.compute_priors_with_scales(image_size, priors_size)
        else:
            mean = self.compute_priors(image_size, priors_size)

        mean = np.asarray(mean).reshape(-1, 4)
        if self.rotations is not None:
            rotations = np.tile(self.rotations, (len(mean), 1)).reshape(-1, 1)
            mean = np.tile(mean, (1, len(self.rotations))).reshape(-1, 4)
            mean = np.hstack((mean, rotations))

        self.priors = torch.from_numpy(mean).to(sources[0])
        # TODO
        #if self.clip:
        #    self.priors.clamp_(max=1, min=0)
        self.priors_size = priors_size
        self.image_size = image_size

        return self.priors

    def nbox_per_layer(self):
        if self.sizes:
            nbox = [len(s) for s in self.sizes]
        elif self.scales:
            nbox = [(1 + len(self.aspect_ratios[i]) * 2) * len(self.scales[i])
                    for i in range(self.n_layer)]
        else:
            nbox = [2 + 2 * len(ar) for ar in self.aspect_ratios]

        if self.rotations is not None:
            nbox = [n * len(self.rotations) for n in nbox]
        return nbox

    def num_points_per_box(self):
        if self.rotations is not None:
            return 5
        return 4

    def compute_priors(self, image_size, priors_size):
        mean = []
        for k in range(len(priors_size)):
            f_k = priors_size[k]
            s_k = self.min_sizes[k]
            s_k_prime = np.sqrt(s_k * (self.max_sizes[k]))
            m = []  # HW, K, 4
            for i in range(f_k[0]):
                for j in range(f_k[1]):
                    a = []  # K, 4
                    # unit center x,y
                    cx = (j + 0.5) / f_k[1] * image_size[1]
                    cy = (i + 0.5) / f_k[0] * image_size[0]

                    # aspect_ratio: 1
                    # rel size: min_size
                    a.append([cx, cy, s_k, s_k])

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    a.append([cx, cy, s_k_prime, s_k_prime])

                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        sqrt_ar = sqrt(ar)
                        l = s_k * sqrt_ar
                        s = s_k / sqrt_ar
                        a.append([cx, cy, l, s])
                        a.append([cx, cy, s, l])
                    m.append(a)
            m = np.asarray(m)
            if self.khw_order:
                m = m.swapaxes(0, 1)  # K, HW, 4
            mean += m.flatten().tolist()
        return mean

    def compute_priors_with_scales(self, image_size, priors_size):
        mean = []
        for k in range(len(priors_size)):
            f_k = priors_size[k]
            s_k_prime = self.min_sizes[k]

            ratios = [(1, 1)]
            for ar in self.aspect_ratios[k]:
                if isinstance(ar, numbers.Number):
                    l = sqrt(ar)
                    s = 1 / l
                    ratios.append([l, s])
                    ratios.append([s, l])
                elif isinstance(ar, collections.Iterable):
                    ratios.append(ar)
                    ratios.append(ar[::-1])
            ratios = np.asarray(ratios)

            x = (np.arange(f_k[1]) + 0.5) / f_k[1] * image_size[1]
            y = (np.arange(f_k[0]) + 0.5) / f_k[0] * image_size[0]
            s = s_k_prime * np.array(self.scales[k])

            cy, cx, s, r = np.meshgrid(y, x, s, range(len(ratios)), indexing='ij')
            wh = s[..., None] * ratios[r]
            m = np.stack([cx, cy, wh[..., 0], wh[..., 1]], axis=-1)  # H, W, Sw, Sh, 4

            if self.khw_order:
                m = m.transpose(2, 3, 0, 1, 4)  # Sw, Sh, H, W, 4
            mean += m.flatten().tolist()
        return mean

    def compute_priors_with_size(self, image_size, priors_size):
        """fully configured size"""
        mean = []
        for k in range(len(priors_size)):
            f_k = priors_size[k]
            s_ks = self.sizes[k]
            m = []  # HW, K, 4
            for i in range(f_k[0]):
                for j in range(f_k[1]):
                    # unit center x,y
                    cx = (j + 0.5) / f_k[1] * image_size[1]
                    cy = (i + 0.5) / f_k[0] * image_size[0]

                    a = [[cx, cy, s_k[0], s_k[1]] for s_k in s_ks]  # K, 4
                    m.append(a)
            m = np.asarray(m)
            if self.khw_order:
                m = m.swapaxes(0, 1)  # K, HW, 4
            mean += m.flatten().tolist()
        return mean


if __name__ == '__main__':
    prior = PriorBox(cfg=dict(min_sizes=[32],
                              scales=[[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]],
                              rotations=np.deg2rad(np.linspace(-30, 30, 3)),
                              aspect_ratios=[[(1.4, 7)]]))
    image_size = (512, 512)
    features = [torch.zeros((1, 4, 4, 1))]
    box = prior((image_size, features))
    print(box)
    print(box.size(), prior.priors_size)