from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['min_dim']
        # print("image size:",self.image_size)
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])  # 6
        # print("num priors:",self.num_priors)
        self.variance = cfg['variance'] or [0.1]   #[0.1, 0.2]
        # print("va:",self.variance)
        self.feature_maps = cfg['feature_maps']  #[38, 19, 10, 5, 3, 1]
        # print("m:",self.feature_maps)
        self.min_sizes = cfg['min_sizes']   # [30, 60, 111, 162, 213, 264]
        self.max_sizes = cfg['max_sizes']   # [60, 111, 162, 213, 264, 315]
        # print("max:",self.max_sizes)
        # print("min:",self.min_sizes)
        self.steps = cfg['steps']          #[8, 16, 32, 64, 100, 300]
        # print("steps:",self.steps)
        self.aspect_ratios = cfg['aspect_ratios']   # [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        # print("aspect ratios:",self.aspect_ratios)
        self.clip = cfg['clip']

        # print("clip:",self.clip)
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # print("self.feature_maps:",self.feature_maps)

        for k, f in enumerate(self.feature_maps):  # features_map大小 38,19,10,5,3,1

            for i, j in product(range(f), repeat=2):

                f_k = self.image_size / self.steps[k]   # steps [8, 16, 32, 64, 100, 300]
                # unit center x,y
                cx = (j + 0.5) / f_k      #得到相对于原图的偏移值 在0～1  之间
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k]/self.image_size   #预测图像的宽度最小的size [30, 60, 111, 162, 213, 264]
                mean += [cx, cy, s_k, s_k]
                # if (f == 5):
                #     number += 2
                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                    mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        # print("number..",number)
        # print("output.size:",output.shape)
        if self.clip:
            output.clamp_(max=1, min=0)

        # print("output size:...", output.shape)
        return output
