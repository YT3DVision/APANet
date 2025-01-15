import argparse
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from basicsr.models.APANet import DerainNet as Baseline
from common.dataset import MyTrainDataSet as train_Dataset
from common.dataset import MyValueDataSet as val_Dataset
from common.ssim import SSIM

# 为CPU\GPU设置种子，保证每次的随机初始化都是相同的，从而保证结果可以复现。
torch.cuda.manual_seed(1234)
torch.manual_seed(1234)
random.seed(1234)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Synthetic')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr_min', type=float, default=2e-6)
    parser.add_argument('--n_epochs', type=int, default=250, help='number of epochs to train')
    parser.add_argument('--train_patch_size', type=int, default=[128, 256])
    parser.add_argument('--train_batch_size', type=int, default=[8, 2])
    parser.add_argument('--train_milestone', type=int, default=[100, 250])
    parser.add_argument('--train_dir', type=str,
                        default=r'../../Dataset/Kitti/K12/training')
    parser.add_argument('--test_dir', type=str,
                        default=r'../../Dataset/Kitti/K12/testing')

    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--val_patch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--val_freq', type=int, default=10)

    parser.add_argument('--ckp_name', type=str, default='model_latest.pth')
    parser.add_argument('--model_dir', type=str, default='./training_model/')
    parser.add_argument('--runs_dir', type=str, default='./runs/')

    return parser.parse_args()


cfg = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def PSNR(img1, img2):
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class Session:
    def __init__(self):
        cudnn.bechmark = True
        self.tb_writer = SummaryWriter(log_dir=os.path.join(cfg.runs_dir, cfg.task))
        self.device = cfg.device
        self.net = Baseline()
        self.ssim_loss = SSIM().cuda()
        self.initial_epoch = 1
        self.n_epochs = cfg.n_epochs
        self.train_patch_size = cfg.train_patch_size
        self.train_batch_size = cfg.train_batch_size
        self.train_milestone = cfg.train_milestone
        self.train_dir = cfg.train_dir
        self.test_dir = cfg.test_dir

        self.optimizer = Adam(self.net.parameters(), lr=cfg.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=cfg.lr_min)
        self.net = torch.nn.DataParallel(self.net).cuda()
        self.task = cfg.task
        self.model_dir = os.path.join(cfg.model_dir, self.task)
        self.best_psnr = None
        self.lr = None
        ensure_dir(self.model_dir)
        if self.n_epochs not in self.train_milestone:
            self.train_milestone.append(self.n_epochs)

    def get_train_dataloader(self, epoch):
        flag = 0
        for mark in self.train_milestone:
            if epoch <= mark:
                break
            else:
                flag += 1
        dataset = train_Dataset(self.train_dir, self.train_patch_size[flag])
        dataloader = DataLoader(dataset, batch_size=self.train_batch_size[flag], shuffle=True,
                                num_workers=cfg.num_workers, drop_last=True, pin_memory=True)

        return dataloader

    def get_val_dataloader(self, ):
        dataset = val_Dataset(self.test_dir, cfg.val_patch_size)
        dataloader = DataLoader(dataset, batch_size=cfg.val_batch_size, shuffle=True, num_workers=cfg.num_workers,
                                drop_last=False, pin_memory=True)
        return dataloader

    def load_checkpoint_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            print('Loading checkpoint %s' % ckp_path)
            ckp = torch.load(ckp_path, map_location='cuda')
        except FileNotFoundError:
            print('No checkpoint %s' % ckp_path)
            return

        self.net.module.load_state_dict(ckp['net'])

        self.initial_epoch = ckp['initial_epoch'] + 1
        for i in range(1, self.initial_epoch):
            self.scheduler.step()
        self.lr = self.optimizer.param_groups[0]['lr']
        self.best_psnr = ckp.get('best_psnr')

        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:%f" % (self.lr))
        print('------------------------------------------------------------------------------')
        if self.best_psnr is not None:
            print("==> Resuming Training with best psnr:%f" % (self.best_psnr))
            print('------------------------------------------------------------------------------')

        print('Continue Train after %d round' % self.initial_epoch)

    def save_checkpoint_net(self, name, epoch):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'initial_epoch': epoch,
            'best_psnr': self.best_psnr,
        }
        torch.save(obj, ckp_path)

    def inf_batch(self, x, y, iters):
        self.net.zero_grad()
        self.optimizer.zero_grad()
        X, Y = x.cuda(), y.cuda()
        pre_y = self.net(X)
        """backward"""
        ssim_loss = 1 - self.ssim_loss(pre_y, Y)
        loss = ssim_loss
        loss.backward()
        self.optimizer.step()

        ssim = (self.ssim_loss(pre_y[:, :3], Y[:, :3]) + self.ssim_loss(pre_y[:, 3:], Y[:, 3:])) / 2
        iters.set_description('Training !!!  Batch Loss %.6f, SSIM %.6f.' % (loss.item(), ssim.item()))
        return loss.data.cpu().numpy(), ssim.data.cpu().numpy()

    def inf_batch_val(self, x, y):
        X, Y = x.cuda(), y.cuda()
        b, c, _, _ = X.shape
        factor = 8
        h, w = X.shape[2], X.shape[3]
        H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
        padh = H - h if h % factor != 0 else 0
        padw = W - w if w % factor != 0 else 0
        X = F.pad(X, (0, padw, 0, padh), 'reflect')
        with torch.no_grad():
            pre_y = self.net(X)
            pre_y = pre_y[:, :, :h, :w]

        ssim = (self.ssim_loss(pre_y[:, :3], Y[:, :3]) + self.ssim_loss(pre_y[:, 3:], Y[:, 3:])) / 2
        psnr = PSNR(pre_y.data.cpu().numpy() * 255, Y.data.cpu().numpy() * 255)
        return ssim.data.cpu(), psnr


def run_train_val():
    sess = Session()
    sess.load_checkpoint_net(cfg.ckp_name)
    dt_val = sess.get_val_dataloader()
    best_epoch = 0
    best_psnr = 0 if sess.best_psnr is None else sess.best_psnr
    best_ssim = 0
    tags = ['Train_Loss', 'Train_SSIM', 'Eval_PSNR', 'Eval_SSIM', 'lr']
    if not os.path.exists('./log_test'):
        os.makedirs('./log_test')
    for epoch in range(sess.initial_epoch, sess.n_epochs + 1):
        dt_train = sess.get_train_dataloader(epoch)
        time_start = time.time()
        loss_all = 0
        SSIM_all = 0
        train_sample = 0
        sess.net.train()
        iters = tqdm(dt_train, file=sys.stdout)
        for idx_iter, (x, y) in enumerate(iters, 0):
            loss, ssim = sess.inf_batch(x, y, iters)
            loss_all += loss
            SSIM_all += ssim

            train_sample += 1
        SSIM = SSIM_all / train_sample
        loss_all = loss_all * dt_train.batch_size

        """Evaluation"""

        if epoch % cfg.val_freq == 0:  # 5个epoch保存一次
            ssim_val = []
            psnr_val = []
            sess.net.eval()
            for i, (x, y) in enumerate(tqdm(dt_val), 0):
                ssim, psnr = sess.inf_batch_val(x, y)
                ssim_val.append(ssim)
                psnr_val.append(psnr)

            ssim_val = torch.stack(ssim_val).mean().item()
            psnr_val = np.stack(psnr_val).mean().item()

            sess.tb_writer.add_scalar(tags[2], psnr_val, epoch)
            sess.tb_writer.add_scalar(tags[3], ssim_val, epoch)
            txt_path = './log_test/' + 'val_' + cfg.task + '.txt'
            if os.path.isfile(txt_path):
                logfile = open(txt_path, 'a+')
            else:
                logfile = open(txt_path, 'w')
            logfile.write(
                'epoch = ' + str(epoch) + '\t'
                                          'ssim  = ' + str(ssim_val) + '\t'
                                                                       'pnsr  = ' + str(psnr_val) + '\t'
                                                                                                    '\n\n'
            )
            logfile.close()
            # 如果psnr_avg大于best_psnr则单独保存
            if psnr_val >= best_psnr:
                best_psnr = psnr_val
                sess.best_psnr = best_psnr
                best_ssim = ssim_val
                best_epoch = epoch

                sess.save_checkpoint_net('best', epoch)
            print("[epoch %d PSNR: %.4f SSIM:%.4f --- best_epoch %d Best_PSNR %.4f Best_SSIM %.4f]" % (
                epoch, psnr_val, ssim_val, best_epoch, best_psnr, best_ssim))
        sess.scheduler.step()

        '''calculation time'''
        print("------------------------------------------------------------------")

        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tSSIM: {:.4f}\tBase_lr {:.8f}".format(epoch,
                                                                                           time.time() - time_start,
                                                                                           loss_all, SSIM,
                                                                                           sess.scheduler.get_last_lr()[
                                                                                               0]))

        print("------------------------------------------------------------------")
        sess.save_checkpoint_net('model_latest.pth', epoch)
        sess.tb_writer.add_scalar(tags[0], loss_all, epoch)
        sess.tb_writer.add_scalar(tags[1], SSIM, epoch)
        sess.tb_writer.add_scalar(tags[4], sess.optimizer.param_groups[0]['lr'], epoch)


if __name__ == '__main__':
    run_train_val()
