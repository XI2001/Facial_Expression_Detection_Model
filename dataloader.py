import pathlib
import os
import cv2
import torch
from torch.utils.data import DataLoader, Dataset
import params
import numpy as np
import random
import albumentations as A
import copy

args = params.parse()
print(args.labelDir)
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Albumentations:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=args.imgsize):
        self.transform = True
        # try:

        T = [
            A.RandomResizedCrop(height=size, width=size, scale=(0.8, 1.2), ratio=(0.9, 1.2), p=0.5),
            A.Blur(p=0.2),
            A.MedianBlur(p=0.2),
            A.ToGray(p=0.05),
            A.CLAHE(p=0.02),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomGamma(p=0.2),
            A.ImageCompression(quality_lower=75, p=0.1),
        ]  # transforms
        self.transform = A.Compose(T, bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

        # except ImportError:  # package not installed, skip
        #     pass
        # except Exception as e:
        #     LOGGER.info(f'{prefix}{e}')

    def __call__(self, im, labels, p=1):
        labels = labels.reshape(1, -1)
        if self.transform and random.random() < p:
            new = self.transform(image=im, bboxes=labels[:, 1:], class_labels=labels[:, 0])  # transformed
            im, labels = new['image'], np.array([[c, *b] for c, b in zip(new['class_labels'], new['bboxes'])]).reshape(
                -1)
        return im, labels


class Albumentations2:
    # YOLOv5 Albumentations class (optional, only used if package is installed)
    def __init__(self, size=args.imgsize):
        self.transform = True
        # try:

        T = [
            A.RandomCrop(height=320, width=320, p=0.5)
        ]  # transforms
        self.transform = A.Compose(T)

        # except ImportError:  # package not installed, skip
        #     pass
        # except Exception as e:
        #     LOGGER.info(f'{prefix}{e}')

    def __call__(self, im, p=1):
        if self.transform and random.random() < p:
            new = self.transform(image=im)  # transformed
            im = new['image']
        return im


def rect(img):
    img_ = img.shape
    shape = np.array(img.shape)
    # 放大的倍数

    shape[0:2] = (shape[0:2] * (args.imgsize / max(shape[0:2]))).astype(int)
    paddingAxis = np.argmin(shape[0:2]).reshape(-1)[0]
    shape[1 - paddingAxis] = args.imgsize  # 防止填充到了639， int一下删掉了。。。。
    img = cv2.resize(img, (shape[1], shape[0]))

    paddingNuml = int((args.imgsize - shape[paddingAxis]) / 2)
    paddingNumr = args.imgsize - paddingNuml - shape[paddingAxis]
    shape_ = np.array([args.imgsize, args.imgsize, 1])
    shape_[paddingAxis] = paddingNuml
    lshape = copy.deepcopy(shape_)
    shape_[paddingAxis] = paddingNumr
    rshape = copy.deepcopy(shape_)

    green = np.array([[[0, 255, 0]]], dtype=np.uint8)
    img = np.concatenate(
        [
            np.repeat(np.repeat(green, lshape[0], axis=0), lshape[1], axis=1),
            img,
            np.repeat(np.repeat(green, rshape[0], axis=0), rshape[1], axis=1)
        ],
        axis=paddingAxis
    ).astype(np.uint8)

    return img


import random


def getDataset(train_size=0.8):
    idxs = [i for i in range(len(os.listdir(args.labelDir)))]
    random.shuffle(idxs)
    size = len(idxs)
    train = idxs[:int(size * train_size)]
    test = idxs[int(size * train_size):]
    return emoDataLoader(dataset=EmoDataSet(train, True)), emoDataLoader(dataset=EmoDataSet(test, False))


class EmoDataSet(Dataset):
    def __init__(self, idx=None, train=True):
        super(EmoDataSet, self).__init__()
        self.labelDataDir = np.array([pathlib.Path(args.labelDir).joinpath(i) for i in os.listdir(args.labelDir)])
        self.imageDataDir = np.array([pathlib.Path(args.imageDir).joinpath(i.stem + '.png') for i in self.labelDataDir])
        self.labelDataDir = self.labelDataDir[idx]
        self.imageDataDir = self.imageDataDir[idx]
        self.alb = Albumentations()
        self.alb2 = Albumentations2()
        self.train = True

    def __len__(self):
        return len(self.labelDataDir)

    def __getitem__(self, item):
        imagePath = self.imageDataDir[item]
        labelPath = self.labelDataDir[item]
        if imagePath.stem != labelPath.stem:
            raise ValueError()
        im = cv2.imread(str(imagePath), flags=cv2.IMREAD_COLOR)
        with open(labelPath) as f:
            labels = np.array(f.read().strip().split(' ')).astype(np.float64)
            finalLabel = labels[0]
        if self.train == True:
            # 图像增广
            im, labels = self.alb(im, labels)
            # 图像裁剪
        h, w = im.shape[:2]
        labels[1::2] = w * labels[1::2]
        labels[2::2] = h * labels[2::2]
        labels_xyxy = np.zeros(labels.shape, dtype=int)
        labels_xyxy[1] = int(np.clip(labels[1] - labels[3] / 2, 0, w))
        labels_xyxy[2] = int(np.clip(labels[2] - labels[4] / 2, 0, h))
        labels_xyxy[3] = int(np.clip(labels[1] + labels[3] / 2, 0, w)) + 1
        labels_xyxy[4] = int(np.clip(labels[2] + labels[4] / 2, 0, h)) + 1
        im = im[labels_xyxy[2]:labels_xyxy[4], labels_xyxy[1]:labels_xyxy[3], :]

        if self.train == True:
            # 随机裁剪
            shape = np.array(im.shape)
            shape[0:2] = (shape[0:2] * (args.imgsize / max(shape[0:2]))).astype(int)
            paddingAxis = np.argmin(shape[0:2]).reshape(-1)[0]
            shape[1 - paddingAxis] = args.imgsize  # 防止填充到了639， int一下删掉了。。。。
            im = cv2.resize(im, (shape[1], shape[0]))
            im = self.alb2(im)

        # 矩形填充
        im = rect(im)

        return im, int(finalLabel)

from tqdm import tqdm
class EmoDetectSet(Dataset):
    def __init__(self):
        super(EmoDetectSet, self).__init__()
        self.labelDataDir = np.array([pathlib.Path(args.detectLabelDir).joinpath(i) for i in tqdm(os.listdir(args.detectLabelDir))])
        self.imageDataDir = np.array([pathlib.Path(args.detectImageDir).joinpath(i.stem + '.jpg') for i in tqdm(self.labelDataDir)])
        # self.labelDataDir = np.array([pathlib.Path(args.detectLabelDir).joinpath(i.stem + '.txt') for i in self.imageDataDir])

    def __len__(self):
        return len(self.labelDataDir)

    def __getitem__(self, item):
        imagePath = self.imageDataDir[item]
        labelPath = self.labelDataDir[item]
        if imagePath.stem != labelPath.stem:
            raise ValueError()
        im = cv2.imread(str(imagePath), flags=cv2.IMREAD_COLOR)
        with open(labelPath) as f:
            labels = np.array(f.readline().strip().split(' ')).astype(np.float64)
            finalLabel = labels[0]

        # 人头切割出来，用yolo识别的时候记得用class筛选一下
        h, w = im.shape[:2]
        labels[1::2] = w * labels[1::2]
        labels[2::2] = h * labels[2::2]
        labels_xyxy = np.zeros(labels.shape, dtype=int)
        labels_xyxy[1] = int(np.clip(labels[1] - labels[3] / 2, 0, w))
        labels_xyxy[2] = int(np.clip(labels[2] - labels[4] / 2, 0, h))
        labels_xyxy[3] = int(np.clip(labels[1] + labels[3] / 2, 0, w)) + 1
        labels_xyxy[4] = int(np.clip(labels[2] + labels[4] / 2, 0, h)) + 1
        im = im[labels_xyxy[2]:labels_xyxy[4], labels_xyxy[1]:labels_xyxy[3], :]

        # 矩形填充
        im = rect(im)

        return im, int(finalLabel), imagePath.absolute()


def emoColFun(datasetIns, grey=args.grey):
    imgs = []
    labels = []
    for datasetIn in datasetIns:
        img, label = datasetIn
        if not grey:
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(int(args.imgsize), int(args.imgsize), -1)
            img_c = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(torch.from_numpy(np.concatenate((img_grey, img_c), axis=2)).float().div(255.0))
        else:
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(int(args.imgsize), int(args.imgsize), -1)
            imgs.append(torch.from_numpy(img_grey).float().div(255.0))
        labels.append(label)
    imgs = torch.stack(imgs, dim=0).to(args.device).permute(0, 3, 1, 2)
    labels = torch.tensor(labels).long().to(args.device)
    return imgs, labels

def emoDetectFun(datasetIns, grey=args.grey):
    imgs = []
    labels = []
    paths = []
    for datasetIn in datasetIns:
        img, label, path_ = datasetIn
        paths.append(path_)
        if not grey:
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(int(args.imgsize), int(args.imgsize), -1)
            img_c = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(torch.from_numpy(np.concatenate((img_grey, img_c), axis=2)).float().div(255.0))
        else:
            img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).reshape(int(args.imgsize), int(args.imgsize), -1)
            imgs.append(torch.from_numpy(img_grey).float().div(255.0))
        labels.append(label)
    imgs = torch.stack(imgs, dim=0).to(args.device).permute(0, 3, 1, 2)
    labels = torch.tensor(labels).long().to(args.device)
    return imgs, labels, paths


def emoDataLoader(dataset=EmoDataSet()):
    return DataLoader(dataset=dataset, batch_size=args.batchSize, collate_fn=emoColFun, shuffle=True)

def emoDetectDataLoader(dataset = EmoDetectSet()):
    return DataLoader(dataset=dataset , batch_size=args.detectBatchSize, collate_fn=emoDetectFun, shuffle=False)

def imageShow(cv2Image):
    cv2.imshow("Image", cv2Image)
    cv2.waitKey()  # 要加的两行代码
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # a,b = getDataset()
    # datal = EmoDataSet([1,2,4],False)
    a = emoDetectDataLoader()
    datal = EmoDetectSet()
    for data in a:
        print(data)
        break
    # im = a[1]
    imageShow(datal[2][0])
