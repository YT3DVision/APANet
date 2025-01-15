import numpy as np
import math
import torch


def PSNR(img1, img2):
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def PSNR_cuda(img1, img2):
    mse = torch.mean((img1 - img2) ** 2).cpu().item()
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
