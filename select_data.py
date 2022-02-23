#!/usr/bin/env python3

import math
import pickle
import random

import torch
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as trans
from tqdm import tqdm

IMAGENET_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_TRANSFORM = trans.Compose(
    # [trans.Scale(256), trans.CenterCrop(224), trans.ToTensor()]  # For VGG-16 and ResNet50
    [trans.Scale(342), trans.CenterCrop(299), trans.ToTensor()]  # For Inception_V3
)


def apply_normalization(imgs, dataset):
    if dataset == "imagenet":
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
    imgs_tensor = imgs.clone()
    if dataset == "mnist":
        imgs_tensor = (imgs_tensor - mean[0]) / std[0]
    else:
        if imgs.dim() == 3:
            for i in range(imgs_tensor.size(0)):
                imgs_tensor[i, :, :] = (imgs_tensor[i, :, :] - mean[i]) / std[i]
        else:
            for i in range(imgs_tensor.size(1)):
                imgs_tensor[:, i, :, :] = (imgs_tensor[:, i, :, :] - mean[i]) / std[i]
    return imgs_tensor


def get_preds(
    model, inputs, dataset_name, correct_class=None, batch_size=50, return_cpu=True
):
    num_batches = int(math.ceil(inputs.size(0) / float(batch_size)))
    softmax = torch.nn.Softmax()
    all_preds, all_probs = None, None
    transform = trans.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    for i in range(num_batches):
        upper = min((i + 1) * batch_size, inputs.size(0))
        input = apply_normalization(inputs[(i * batch_size) : upper], dataset_name)
        input_var = torch.autograd.Variable(input.cuda(), volatile=True)
        output = softmax.forward(model.forward(input_var))
        if correct_class is None:
            prob, pred = output.max(1)
        else:
            prob, pred = (
                output[:, correct_class],
                torch.autograd.Variable(torch.ones(output.size()) * correct_class),
            )
        if return_cpu:
            prob = prob.data.cpu()
            pred = pred.data.cpu()
        else:
            prob = prob.data
            pred = pred.data
        if i == 0:
            all_probs = prob
            all_preds = pred
        else:
            all_probs = torch.cat((all_probs, prob), 0)
            all_preds = torch.cat((all_preds, pred), 0)
    return all_preds, all_probs


testset = datasets.ImageFolder("ImageNet2012/val", IMAGENET_TRANSFORM)


# Save image indices for every model
model = models.inception_v3(pretrained=True).cuda()  # Change this for different models
# model = models.vgg16(pretrained=True).cuda()
# model = models.resnet50(pretrained=True).cuda()
model.eval()

with torch.no_grad():
    ids = []
    for i in tqdm(range(len(testset))):
        image, label = testset[i]
        image = image.unsqueeze(0)
        pred, _ = get_preds(model, image, "imagenet", batch_size=1)
        if pred.item() == label:
            ids.append(i)
correct_indices = open("./inception_v3", "w")  # Change this for different models
# correct_indices = open("./vgg16", "w")
# correct_indices = open("./resnet50", "w")
for i in ids:
    correct_indices.write(f"{i}\n")

# Final result
files = ["resnet50", "inception_v3", "vgg16"]
correct_indices = []

for f in files:
    indices = open(f, "r").readlines()
    indices = [int(v.strip()) for v in indices]
    imgs = []
    for ii in indices:
        imgs.append(testset.imgs[ii])
    # pickle.dump(a, open(f + ".pkl", "wb"))
    correct_indices.append(set(imgs))

# Images classified correctly by all models
inter = (
    correct_indices[0].intersection(correct_indices[1]).intersection(correct_indices[2])
)

c2f = [[] for i in range(1000)]
for ii in inter:
    c2f[ii[1]].append(ii[0])


results = []
# Sample randomly
for i in range(1000):
    s = random.sample(c2f[i], 1)[0]
    results.append([s, str(i)])
    v = random.randint(0, 999)
    while v == i:
        v = random.randint(0, 999)
    s2 = random.sample(c2f[v], 1)[0]
    results[-1].append(s2)
    results[-1].append(str(v))
results = [",".join(v) + "\n" for v in results]
f = open("list.txt", "w")
f.writelines(results)
