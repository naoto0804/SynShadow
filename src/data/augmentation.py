import numpy as np
import torch
import torch.nn.functional as F


class MaskAugmentor():
    def __init__(self, p_min=-8, p_max=8, use_gpu=True):
        """
        Perform random dilation/erosion using convolution uniformly on a mask with the size (1, 1, H, W)
        If it's erosion, invert mask before and after convolution
        Args:
            p_min (int): minimum padding for dilation, if smaller than zero, its eroding
            p_max (int): maximum padding for dilation
            use_gpu (bool): convolution on GPU, used to avoid error in MKL-DNN on CPU
        """
        self.p_min = p_min
        self.p_max = p_max
        self.weights = {}
        # if use_gpu:
        #     self.device = torch.device('cuda')
        # else:
        self.device = torch.device('cpu')

        for pad in range(p_min, p_max + 1):
            kernel = 2 * abs(pad) + 1
            weight_size = (1, 1, kernel, kernel)
            weight = torch.ones(weight_size)
            if use_gpu:
                weight = weight.to(self.device)
            self.weights[pad] = weight

    def __call__(self, mask):
        mask = mask.unsqueeze(0)
        mask = mask.float().to(self.device)

        pad = np.random.randint(self.p_min, self.p_max + 1)
        if pad < 0: mask = 1.0 - mask
        with torch.no_grad():
            output = F.conv2d(mask, self.weights[pad], padding=abs(pad)) > 0
            output = output.float().cpu()
        if pad < 0: output = 1.0 - output
        return output.squeeze(0)
