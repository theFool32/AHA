import math
import os
from math import isinf, isnan

import cv2
import numpy as np
import torch as ch
from numpy.core.defchararray import mod
from scipy.fftpack import dct, idct
from torch import gt
from torchvision.utils import save_image
from tqdm import tqdm
import ipdb

ch.manual_seed(1234)
ch.cuda.manual_seed(1234)
np.random.seed(1234)

def sample_gaussian_torch(image_size, dct_ratio=1.0):
    x = ch.zeros(image_size, device="cuda")
    fill_size = int(image_size[-1] * dct_ratio)
    x[:, :, :fill_size, :fill_size] = ch.randn(x.size(0), x.size(1), fill_size, fill_size, device="cuda")
    if dct_ratio < 1.0:
        x = ch.from_numpy(idct(idct(x.cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho')).cuda()
    return x.squeeze().contiguous()


def attack(model, src_img, tgt_img, img_index, image_size):
    alpha = 1e-2 * ch.ones(src_img.size(0), device="cuda")
    beta = 1e-2
    m = 1e-2
    stats_adversarial = []
    stats = [[] for ii in range(src_img.size(0))]
    alphas = []
    alphas.append(alpha)
    muls = []
    labels = []

    def get_norm(t):
        return (t.view(t.size(0), -1)**2).sum(dim=1).sqrt()

    def log(dis, query_time=None):
        if query_time is None:
            query_time = model.call_time
        for ii in range(src_img.size(0)):
            stats[ii].append((query_time, dis[ii].item()))

    def make_adv(adv):
        return adv

    def test(adv):
        adv = make_adv(adv)
        return model.forward(adv)

    def calc_norm(adv):
        d = make_adv(adv) - src_img
        return get_norm(d)

    def to_boundary(img):
        high, low = img.clone(), src_img.clone()
        while (high - low).abs().max() > 1e-5:
            mid = (high + low) / 2
            index = test(mid)
            high[index==1] = mid[index==1]
            low[index==0] = mid[index==0]
        return high

    DIM = int(image_size / 4)

    x_adv = tgt_img.clone()
    log(calc_norm(x_adv), 0)
    orig_dist = calc_norm(x_adv)

    x_adv = to_boundary(x_adv)
    log(calc_norm(x_adv))
    bias = ch.zeros(x_adv.shape[0], 3, DIM, DIM, device="cuda")

    best_dist = calc_norm(x_adv)
    step = 0

    while True:
        step += 1
        if model.call_time >= 20000:
            return stats

        to_orig_direction = src_img - make_adv(x_adv)
        od_norm = get_norm(to_orig_direction)

        eta = sample_gaussian_torch(bias.size())
        z = to_orig_direction.abs()
        z = ch.nn.functional.interpolate(z, (DIM,DIM))
        eta = eta * z + bias

        if DIM < image_size:
            eta_img = ch.nn.functional.interpolate(
                eta, (image_size, image_size)
            )
        else:
            eta_img = eta


        new_x_adv = x_adv + alpha.view(-1,1,1,1) * to_orig_direction + beta * od_norm.view(-1, 1,1,1) * eta_img / get_norm(eta_img).view(-1,1,1,1)
        new_x_adv = ch.clamp(new_x_adv, 0, 1)

        new_label = test(new_x_adv)
        stats_adversarial.append(new_label)
        dist = calc_norm(new_x_adv)

        x_adv[new_label==1] = new_x_adv[new_label==1]
        bias = (1-m) * bias + m * eta * (new_label*2-1).view(new_label.size(0),1,1,1)
        best_dist[new_label==1] = ch.min(best_dist[new_label==1], dist[new_label==1])
        log(best_dist)
        labels.append(new_label)
        print(f"{img_index}/{step}/{model.call_time}: {dist.mean()}/{best_dist.mean()} {alpha.mean()}")

        expect_dist = alpha*od_norm + beta / get_norm(eta_img) * (eta_img.view(eta_img.size(0), -1) * to_orig_direction.view(to_orig_direction.size(0), -1)).sum(1)
        actual_dist = od_norm - calc_norm(x_adv)
        expect_dist = expect_dist.abs()
        actual_dist = actual_dist.abs()
        mul = ch.min(expect_dist, actual_dist) / (ch.max(expect_dist, actual_dist)+1e-12)
        muls.append(mul)

        if len(muls) == 30:
            m_mul = sum(muls) / len(muls)
            alpha = alpha * ((m_mul+0.8).pow(2))
            muls.clear()
            alpha[alpha < 1e-9] = 1e-2
