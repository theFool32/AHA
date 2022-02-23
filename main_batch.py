import argparse
import os
import sys
import time

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from sklearn.metrics import auc

from attack_batch import attack

trans = transforms

#torch.set_default_tensor_type('torch.cuda.FloatTensor')


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


def distance(feature1, feature2):
    norm = torch.norm(feature1 - feature2)
    norm = norm ** 2
    return norm


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5

IMAGENET_TRANSFORM = trans.Compose(
    [trans.Scale(256), trans.CenterCrop(224), trans.ToTensor()]
)
IMAGENET_TRANSFORM_INCEPTION = trans.Compose(
    [trans.Scale(342), trans.CenterCrop(299), trans.ToTensor()]
)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=['resnet50', 'vgg16', 'inception_v3'])
args = parser.parse_args()
model = args.model

if args.model == 'vgg16':
    fmodel = models.vgg16(pretrained=True).cuda()
elif args.model == 'resnet50':
    fmodel = models.resnet50(pretrained=True).cuda()
elif args.model == 'inception_v3':
    fmodel = models.inception_v3(pretrained=True).cuda()
fmodel = fmodel.eval()

mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))).cuda()
std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))).cuda()

class Model:
    def __init__(self):
        self.call_time = 0
        self.model = fmodel
        self.tgt_label = None

    def set_tgt(self, tgt):
        self.tgt_label = self.get_predict(tgt)
        self.call_time = 0

    def get_predict(self, x):
        # if not isinstance(input, torch.Tensor):
        #     x = torch.from_numpy(x).float()
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        x = (x-mean)/std
        x = x.float()
        pred = self.model(x.cuda())
        # self.call_time += x.shape[0]
        self.call_time += 1
        return pred.argmax(1)

    def forward(self, x):
        return (self.get_predict(x) == self.tgt_label).float()

    def __call__(self, x):
        return (self.get_predict(x) == self.tgt_label).float()


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def load_img(path, transform):
    img = pil_loader(path)
    img = transform(img)
    # img = img.numpy().transpose(1, 2, 0) * 255
    # img = img.numpy()
    return img

def main():
    q_ls = []
    model = Model()
    q_ls=[]
    i = 0
    datas = []
    batch_size = 100
    for line in tqdm(open("list.txt", "r")):
        i += 1
        l = line.strip().split(',')
        source_name = l[0]
        target_name = l[2]
        datas.append((source_name, target_name))

    for i in range(0, len(datas), batch_size):
        img1s = []
        img2s = []
        for ii in range(i, min(i+batch_size, len(datas))):
            source_name, target_name = datas[ii]
            if args.model == 'inception_v3':
                img1 = load_img(source_name, IMAGENET_TRANSFORM_INCEPTION)
                img2 = load_img(target_name, IMAGENET_TRANSFORM_INCEPTION)
                image_size = 299
            else:
                img1 = load_img(source_name, IMAGENET_TRANSFORM)
                img2 = load_img(target_name, IMAGENET_TRANSFORM)
                image_size = 224
            img1s.append(img1)
            img2s.append(img2)
        img1s = torch.stack(img1s).cuda()
        img2s = torch.stack(img2s).cuda()

        model.set_tgt(img2s)

        result = attack(model, img1s, img2s, i, image_size)
        q_ls.append(result)

    _len = len(q_ls)
    np_q_ls = np.zeros((_len, 20001))
    for _i in range(_len):
        for _j in range(len(q_ls[_i])):
            if _j < len(q_ls[_i]) - 1:
                np_q_ls[_i][q_ls[_i][_j][0] : q_ls[_i][_j + 1][0]] = q_ls[_i][_j][1]
            else:
                np_q_ls[_i][q_ls[_i][_j][0] :] = q_ls[_i][_j][1]

    # # m = np_q_ls.mean(0)
    # # x = np.arange(0, 20001)
    # # _auc = round(auc(x, m), 2)
    # # print(round(m[-1], 3), _auc)
    # # plt.plot(m)

    current_time = time.ctime()
    np.save(f"{args.model}_total_{current_time}.npy", np_q_ls)


if __name__ == "__main__":
    with torch.no_grad():
        main()
