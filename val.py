## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
import argparse
import math
import os
from glob import glob

parser = argparse.ArgumentParser(description='Image Deraining')
parser.add_argument('--input_dir', default=r'../', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./Results/', type=str, help='Directory for results')
parser.add_argument('--weights', default=r'./training_model/RainKITTI2012/best', type=str, help='Path to weights')

datasets = ['RainKITTI2012']

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import numpy as np
import torch
import torch.nn.functional as F
from natsort import natsorted
from skimage import img_as_ubyte
from tqdm import tqdm

from basicsr.models.APANet import DerainNet as Baseline
from common import utils

def splitimage(imgtensor, crop_size=128, overlap_size=20):
    _, C, H, W = imgtensor.shape
    hstarts = [x for x in range(0, H, crop_size - overlap_size)]
    while len(hstarts) != 0 and hstarts[-1] + crop_size >= H:
        hstarts.pop()
    hstarts.append(H - crop_size)
    wstarts = [x for x in range(0, W, crop_size - overlap_size)]
    while len(wstarts) != 0 and wstarts[-1] + crop_size >= W:
        wstarts.pop()
    wstarts.append(W - crop_size)
    starts = []
    split_data = []
    for hs in hstarts:
        for ws in wstarts:
            cimgdata = imgtensor[:, :, hs:hs + crop_size, ws:ws + crop_size]
            starts.append((hs, ws))
            split_data.append(cimgdata)
    return split_data, starts


def get_scoremap(H, W, C, B=1, is_mean=True):
    center_h = H / 2
    center_w = W / 2

    score = torch.ones((B, C, H, W))
    if not is_mean:
        for h in range(H):
            for w in range(W):
                score[:, :, h, w] = 1.0 / (math.sqrt((h - center_h) ** 2 + (w - center_w) ** 2 + 1e-3))
    return score


def mergeimage(split_data, starts, crop_size=128, resolution=(1, 3, 128, 128)):
    B, C, H, W = resolution[0], resolution[1], resolution[2], resolution[3]
    tot_score = torch.zeros((B, C, H, W))
    merge_img = torch.zeros((B, C, H, W))
    scoremap = get_scoremap(crop_size, crop_size, C, B=B, is_mean=False)
    for simg, cstart in zip(split_data, starts):
        hs, ws = cstart
        merge_img[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap * simg
        tot_score[:, :, hs:hs + crop_size, ws:ws + crop_size] += scoremap
    merge_img = merge_img / tot_score
    return merge_img


args = parser.parse_args()

##########################
device = 'cuda'
model_restoration = Baseline().to(device)
checkpoint = torch.load(args.weights, map_location=device)
model_restoration.load_state_dict(checkpoint['net'])
print("===>Testing using weights: ", args.weights)
print(checkpoint['initial_epoch'] + 1)
model_restoration.eval()
crop_size = 320
overlap_size = 60
factor = 32


for dataset in datasets:
    result_dir = os.path.join(args.result_dir, dataset)
    os.makedirs(result_dir + r'/left/', exist_ok=True)
    os.makedirs(result_dir + r'/right/', exist_ok=True)

    inp_dir_left = os.path.join(args.input_dir, dataset, r'testing/rain_left/')
    inp_dir_right = os.path.join(args.input_dir, dataset, r'testing/rain_right/')
    files_left = natsorted(glob(os.path.join(inp_dir_left, '*.png')) + glob(os.path.join(inp_dir_left, '*.jpg')))
    files_right = natsorted(glob(os.path.join(inp_dir_right, '*.png')) + glob(os.path.join(inp_dir_right, '*.jpg')))
    
    with torch.no_grad():
        for left_, right_ in tqdm(list(zip(files_left, files_right))):

            if os.path.exists(os.path.join(result_dir + r'/left/', os.path.splitext(os.path.split(left_)[-1])[0] + '.png')):
                continue
            left = np.float32(utils.load_img(left_)) / 255.
            
            left = torch.from_numpy(left).permute(2, 0, 1)
            left = left.unsqueeze(0).to(device)
            right = np.float32(utils.load_img(right_)) / 255.
            right = torch.from_numpy(right).permute(2, 0, 1)
            right = right.unsqueeze(0).to(device)
            input_ = torch.concat([left, right], dim=1)

            # Padding in case images are not multiples of 8
            b, c, h, w = input_.shape
            H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
            padh = H - h if h % factor != 0 else 0
            padw = W - w if w % factor != 0 else 0
            input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
            split_data, starts = splitimage(input_, crop_size=crop_size, overlap_size=overlap_size)
            for i, data in enumerate(split_data):
                split_data[i] = model_restoration(data).cpu()
            restored = mergeimage(split_data, starts, crop_size=crop_size, resolution=(b, c, H, W))

            # Unpad images to original dimensions
            restored = restored[:, :, :h, :w]

            left = torch.clamp(restored[:, :3], 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            right = torch.clamp(restored[:, 3:], 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            utils.save_img(
                (os.path.join(result_dir + r'/left/', os.path.splitext(os.path.split(left_)[-1])[0] + '.png')),
                img_as_ubyte(left))
            utils.save_img(
                (os.path.join(result_dir + r'/right/', os.path.splitext(os.path.split(right_)[-1])[0] + '.png')),
                img_as_ubyte(right))
