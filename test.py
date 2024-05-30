# coding: utf-8
import os

import numpy as np
import torch
from torch import nn
from torchvision import transforms

from tools.config import TEST_SOTS_ROOT, OHAZE_ROOT, HazeRD_ROOT, HazeFD_ROOT
from tools.utils import AvgMeter, check_mkdir, sliding_forward
from model import DM2FNet, DM2FNet_woPhy
from datasets import SotsDataset, OHazeDataset
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from skimage.color import deltaE_ciede2000, lab2rgb, rgb2lab
import time

import argparse
parser = argparse.ArgumentParser(description="Process some experiment settings.")

# 添加 exp_name 参数
parser.add_argument('--exp_name', type=str, required=True, help='The name of the experiment')

# 添加 snapshot 参数
parser.add_argument('--snapshot', type=str, required=True, help='The snapshot info of the model')

# 解析命令行参数
args = parser.parse_args()

# 使用参数
print(f"Experiment Name: {args.exp_name}")
print(f"Snapshot Info: {args.snapshot}")

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

torch.manual_seed(2018)
# torch.cuda.set_device(0)

ckpt_path = './ckpt'
exp_name = args.exp_name
# exp_name = 'O-Haze-20'

# args = {
#     # 'snapshot': 'iter_40000_loss_0.01230_lr_0.000000',
#     # 'snapshot': 'iter_19000_loss_0.04261_lr_0.000014',
#     'snapshot': 'iter_20000_loss_0.04685_lr_0.000000',
# }

to_test = {
    # 'SOTS': TEST_SOTS_ROOT,
    # 'O-Haze': OHAZE_ROOT,
    # 'HazeRD': HazeRD_ROOT,
    'HazeFD': HazeFD_ROOT,
}

to_pil = transforms.ToPILImage()


def main():
    with torch.no_grad():
        criterion = nn.L1Loss().cuda()

        for name, root in to_test.items():
            # if 'SOTS' in name:
            #     net = DM2FNet().cuda()
            #     dataset = SotsDataset(root)
            # elif 'Haze' in name:
            #     net = DM2FNet_woPhy().cuda()
            #     dataset = OHazeDataset(root, 'test')
            # else:
            #     raise NotImplementedError
            net = DM2FNet().cuda()
            dataset = OHazeDataset(root, 'test')

            # net = nn.DataParallel(net)

            if len(args.snapshot) > 0:
                print('load snapshot \'%s\' for testing' % args.snapshot)
                net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args.snapshot + '.pth')))

            net.eval()
            dataloader = DataLoader(dataset, batch_size=1)

            psnrs, ssims, mses, ciede2000s, times = [], [], [], [], []
            loss_record = AvgMeter()

            for idx, data in enumerate(dataloader):
                # import pdb;pdb.set_trace()
                start_time = time.time()  # Start timer
                # haze_image, _, _, _, fs = data
                haze, gts, fs = data
                # print(haze.shape, gts.shape)

                check_mkdir(os.path.join(ckpt_path, exp_name,
                                         '(%s) %s_%s' % (exp_name, name, args.snapshot)))

                haze = haze.cuda()

                if 'O-Haze' in name:
                    res = sliding_forward(net, haze).detach()
                else:
                    res = net(haze).detach()

                loss = criterion(res, gts.cuda())
                loss_record.update(loss.item(), haze.size(0))

                elapsed_time = time.time() - start_time  # Time taken for processing
                times.append(elapsed_time)

                for i in range(len(fs)):
                    r = res[i].cpu().numpy().transpose([1, 2, 0])
                    gt = gts[i].cpu().numpy().transpose([1, 2, 0])

                    # Calculate color difference in CIEDE2000
                    lab_r = rgb2lab(r)
                    lab_gt = rgb2lab(gt)
                    ciede = np.mean(deltaE_ciede2000(lab_gt, lab_r))
                    ciede2000s.append(ciede)

                    mse = mean_squared_error(gt, r)
                    mses.append(mse)

                    psnr = peak_signal_noise_ratio(gt, r)
                    psnrs.append(psnr)
                    ssim = structural_similarity(gt, r, data_range=1, multichannel=True,
                                                 gaussian_weights=True, sigma=1.5, use_sample_covariance=False, channel_axis=2)
                    ssims.append(ssim)
                    print('predicting for {} ({}/{}) [{}]: PSNR {:.4f}, SSIM {:.4f}, MSE {:.4f}, CIEDE2000 {:.4f}, Time {:.4f}s'
                          .format(name, idx + 1, len(dataloader), fs[i], psnr, ssim, mse, ciede, elapsed_time))



                for r, f in zip(res.cpu(), fs):
                    to_pil(r).save(
                        os.path.join(ckpt_path, exp_name,
                                     '(%s) %s_%s' % (exp_name, name, args.snapshot), '%s.png' % f))

            print(f"[{name}] L1: {loss_record.avg:.6f}, PSNR: {np.mean(psnrs):.6f}, SSIM: {np.mean(ssims):.6f}, MSE: {np.mean(mses):.6f}, CIEDE2000: {np.mean(ciede2000s):.6f}, Avg Time: {np.mean(times):.4f}s")


if __name__ == '__main__':
    main()
