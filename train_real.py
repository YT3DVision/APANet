import argparse
import math
import os
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='Real')
    parser.add_argument('--device', type=str, default='1')
    parser.add_argument('--lr', type=float, default=3e-4, )
    parser.add_argument('--lr_min', type=float, default=2e-6, )
    parser.add_argument('--n_epochs', type=int, default=250, help='number of epochs to train')
    parser.add_argument('--train_patch_size', type=int, default=[128, 256])
    parser.add_argument('--train_batch_size', type=int, default=[8, 2])
    parser.add_argument('--train_milestone', type=int, default=[100, 250])
    parser.add_argument('--train_dir', type=str,
                        default=r'../Kitti/K12/training')
    parser.add_argument('--test_dir', type=str,
                        default=r'../Kitti/K12/testing')
    parser.add_argument('--real_train_dir', type=str,
                        default=r'../StereoRealRain/train/')
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=10)

    parser.add_argument('--ckp_name', type=str, default='sup_latest')
    parser.add_argument('--teacher_ckp_name', type=str, default='teacher.pth')
    parser.add_argument('--model_dir', type=str, default='./training_model/')
    parser.add_argument('--runs_dir', type=str, default='./runs/')

    return parser.parse_args()


cfg = parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device

import numpy as np
import torch
import torch.nn.functional as F
from skimage import img_as_ubyte
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from basicsr.models.APANet import DerainNet as Baseline
from common import utils
from common.dataset import MyTrainDataSet as train_Dataset
from common.dataset import MyValueDataSet as val_Dataset
from common.dataset import RealDataSet as real_Dataset
from common.ssim import SSIM
from common.loss import TVLoss, DarkCLoss, ContrastLoss

"""?CPU\GPU????????????????????????????????"""
torch.cuda.manual_seed(66)
torch.manual_seed(66)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


def PSNR(img1, img2):
    # b, _, _, _ = img1.shape
    img1 = np.clip(img1, 0, 255)
    img2 = np.clip(img2, 0, 255)
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class Session:
    def __init__(self):

        self.lr = None
        self.best_psnr = None
        self.tb_writer = SummaryWriter(log_dir=os.path.join(cfg.runs_dir, cfg.task))
        self.device = cfg.device
        self.net = Baseline().cuda()
        self.teacher = Baseline().cuda()
        self.ssim_loss = SSIM().cuda()
        layer_weights = {'relu1_1': 1. / 32, 'relu2_1': 1. / 16, 'relu3_1': 1. / 8, 'relu4_1': 1. / 4, 'relu5_1': 1.0}
        self.cr_loss = ContrastLoss(layer_weights).cuda()
        self.tvloss = TVLoss().cuda()
        self.dcloss = DarkCLoss().cuda()

        self.initial_epoch = 1
        self.n_epochs = cfg.n_epochs
        self.train_patch_size = cfg.train_patch_size
        self.train_batch_size = cfg.train_batch_size
        self.train_milestone = cfg.train_milestone
        self.train_dir = cfg.train_dir
        self.test_dir = cfg.test_dir
        self.real_train_dir = cfg.real_train_dir

        self.optimizer = Adam(self.net.parameters(), lr=cfg.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.n_epochs, eta_min=cfg.lr_min)
        self.task = cfg.task
        self.model_dir = os.path.join(cfg.model_dir, self.task)
        ensure_dir(self.model_dir)

        self.teacher.eval()
        teacher_ckp_path = os.path.join(self.model_dir, cfg.teacher_ckp_name)
        ckp = torch.load(teacher_ckp_path, map_location='cuda:0')
        self.teacher.load_state_dict(ckp['net'])
        for name, para in self.teacher.named_parameters():
            para.requires_grad_(False)

    def get_train_dataloader(self, epoch):
        flag = 0
        for mark in self.train_milestone:
            if epoch <= mark:
                break
            else:
                flag += 1

        dataset_real = real_Dataset(self.real_train_dir, self.train_patch_size[flag])
        dataloader_real = DataLoader(dataset_real, batch_size=self.train_batch_size[flag], shuffle=True,
                                     num_workers=cfg.num_workers, drop_last=True)
        dataset = train_Dataset(self.train_dir, self.train_patch_size[flag], len(dataset_real))
        dataloader = DataLoader(dataset, batch_size=self.train_batch_size[flag], shuffle=True,
                                num_workers=cfg.num_workers, drop_last=True, pin_memory=True)
        return dataloader, dataloader_real

    def get_val_dataloader(self, ):
        dataset = val_Dataset(self.test_dir)
        dataloader = DataLoader(dataset, batch_size=cfg.val_batch_size, shuffle=True, num_workers=cfg.num_workers,
                                drop_last=False)
        return dataloader

    def load_checkpoint_net(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            print('Loading checkpoint %s' % ckp_path)
            ckp = torch.load(ckp_path, map_location='cuda:0')
        except FileNotFoundError:
            print('No checkpoint %s' % ckp_path)
            return

        self.net.load_state_dict(ckp['net'])

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
            'net': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'initial_epoch': epoch,
            'best_psnr': self.best_psnr,
        }
        torch.save(obj, ckp_path)

    def inf_batch(self, x, y, z, iters, idx_iter):
        # Synthetic scenes
        self.net.zero_grad()
        self.optimizer.zero_grad()
        X, Y, Z = x.cuda(), y.cuda(), z.cuda()
        pre_y = self.net(X)
        ssim_loss = 1 - self.ssim_loss(pre_y, Y)

        # Real-world scenes
        pre_z = self.net(Z)
        pre_gt = self.teacher(Z)
        hcr_loss = (self.cr_loss(pre_z[:, :3], pre_gt[:, :3], Z[:, :3]) + self.cr_loss(pre_z[:, 3:], pre_gt[:, 3:],
                                                                                       Z[:, 3:])) * 0.5
        tvloss = self.tvloss(pre_z)
        dcloss = self.dcloss(pre_z)
        loss_real = ssim_loss + tvloss * 5e-5 + dcloss * 1e-6 + hcr_loss * 2e-3
        loss_real.backward()
        self.optimizer.step()

        ssim = self.ssim_loss(pre_y, Y)
        iters.set_description(
            'Training !!! SSIM %.6f, tv_loss %.6f, dc_loss %.6f, hcr_loss %.6f.' % (
                ssim.item(), tvloss.item(), dcloss.item(), hcr_loss.item()))
        return loss_real.data.cpu().numpy(), ssim.data.cpu().numpy()

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
        dt_train, dt_train_real = sess.get_train_dataloader(epoch)
        time_start = time.time()
        loss_all = 0
        SSIM_all = 0
        train_sample = 0
        sess.net.train()
        iters = tqdm(dt_train, file=sys.stdout)
        iters_real = iter(dt_train_real)
        for idx_iter, (x, y) in enumerate(iters, 0):
            z = next(iters_real)
            loss, ssim = sess.inf_batch(x, y, z, iters, idx_iter)
            loss_all += loss
            SSIM_all += ssim
            train_sample += 1
        SSIM = SSIM_all / train_sample
        loss_all = loss_all * dt_train.batch_size

        """Evaluation"""

        if epoch % cfg.val_freq == 0:  # 5?epoch????
            ssim_val = []
            psnr_val = []
            sess.net.eval()
            # sess.save_checkpoint_net('net_%s_epoch' % str(epoch+1),epoch+1)
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

            # ??psnr_avg??best_psnr?????
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
        sess.save_checkpoint_net('model_latest_' + str(epoch) + '.pth', epoch)
        sess.tb_writer.add_scalar(tags[0], loss_all, epoch)
        sess.tb_writer.add_scalar(tags[1], SSIM, epoch)
        sess.tb_writer.add_scalar(tags[4], sess.optimizer.param_groups[0]['lr'], epoch)


if __name__ == '__main__':
    run_train_val()
