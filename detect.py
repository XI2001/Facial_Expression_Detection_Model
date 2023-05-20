import pathlib

import torch.nn
import os
import model
import params
import dataloader
from model import YOLOV5, resnet50
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
import pandas as pd
import numpy as np

EmodetectData = dataloader.emoDetectDataLoader()
# net = YOLOV5()
net = torch.load('weights/testAccBest.pt')
net = net.to('cuda')
net.eval()
loss = torch.nn.CrossEntropyLoss()


def calPre(x, label):
    _ = torch.softmax(x, 1)
    # print(torch.argmax(x,dim=1))
    precision = torch.sum(torch.abs(torch.argmax(_, dim=1) - label) < 1e-4) / label.size()[0]
    probability = _
    type = torch.argmax(_, dim=1)
    return precision.cpu().numpy(), probability.cpu().numpy(), type.cpu().numpy()


class Decoder():
    def __init__(self):
        self.strLabel = {
            0: "neutral",
            1: "anger",
            2: "contempt",
            3: "disgust",
            4: "fear",
            5: "happy",
            6: "sadness",
            7: "surprise"
        }

    def getStrLabel(self, intlabel):
        _ = []
        for label in intlabel:
            _.append(self.strLabel[label])
        return _


decoder = Decoder()


# 产生对应的检测表
basename = 'exp'
dd = []
for name_ in os.listdir(r'run'):
    dd.append(int(pathlib.Path(name_).stem.split('_')[1]))
if len(dd) > 0:
    basename = basename + '_' + str(max(dd) + 1) + '.csv'
else:
    basename = 'exp_0.csv'


_ = []
with torch.no_grad():
    with autocast():
        for i, (x, label, file) in tqdm(enumerate(EmodetectData), total=len(EmodetectData)):
            x = net(x)
            # 如果detect的label是yolo输出的值，就不用管precision的值，这里的precision是用来做validating的
            precision, probability, type = calPre(x, label)
            strType = decoder.getStrLabel(type)
            # print(precision, type)
            _.append(np.concatenate(
                (np.array(file).reshape(-1, 1), np.array(strType).reshape(-1, 1), type.reshape(-1, 1), probability),
                axis=1))


            if i % 50 == 10:
                df = pd.DataFrame(data=np.concatenate(_, axis=0),
                                  columns=["File",  "FaicalEmotionType", "FaicalEmotionTypeIndex", "neutral", "anger", "contempt", "disgust", "fear", "happy",
                                           "sadness",
                                           "surprise"]).to_csv(r'run/{}'.format(basename), index=False)


df = pd.DataFrame(data=np.concatenate(_, axis=0),
                                  columns=["File",  "FaicalEmotionType", "FaicalEmotionTypeIndex", "neutral", "anger", "contempt", "disgust", "fear", "happy",
                                           "sadness",
                                           "surprise"]).to_csv(r'run/{}'.format(basename), index=False)