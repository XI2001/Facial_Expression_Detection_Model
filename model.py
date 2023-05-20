import torch
import torch.nn as nn

import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
import params
import yaml
import torchvision
args = params.parse()

def autopad(k, p=None, d=1):  # kernel, padding, dilation(扩大)
    # Pad to 'same' shape outputs
    # 为了自动卷积或者自动池化进行自动的扩充
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channel of the middle layer
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                 e=0.5):  # ch_in, ch_out, number_of_CSP, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv2(x)
        y2 = self.cv3(self.m(self.cv1(x)))
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        for _ in k:
            if _ % 2 == 0:
                raise ValueError('传入的kernel_size应该是奇数')
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=kernelsize, stride=1, padding=kernelsize // 2) for kernelsize in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], dim=1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1):
        super(Focus, self).__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat(
            [
                x[..., ::2, ::2],
                x[..., 1::2, ::2],
                x[..., ::2, 1::2],
                x[..., 1::2, 1::2]
            ], dim=1
        ))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    def __init__(self, dim=1):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, x):
        return torch.concat(x, dim=self.dim)


class Detect(nn.Module):
    """
    @:arg
        nc:
            number of classes
        anchors:
            shape:(nl, na, 2)
            data: config.anchor / config.stride
        ch:
            forward中x[i]的channel就是ch[i]的值

    @:foward
        x:
            dtype: list, length: number of layer for detection
        x[i]:
            tensor, shape (batch_size, )

    @:preparation:
        stride:
            stride[i]对应的是x[i]所对应layer的输入的数据 ，是原始数据多少倍的下采样
            需要进行更改
    """
    export = False  # export mode
    dynamic = False  # force grid reconstruction

    def __init__(self, nc=params.parse().nc, channels=()):
        super(Detect, self).__init__()
        self.c1, self.c2, self.c3 = channels
        self.conv1 = nn.Conv2d(in_channels=self.c1, out_channels=self.c2, kernel_size=3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(in_channels=self.c2, out_channels=self.c3, kernel_size=3, padding=1, stride=2)
        self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=3 *  self.c3, out_channels=self.c3, kernel_size=3, padding=1, stride=2)
        self.conv4 = nn.Conv2d(in_channels=self.c3, out_channels=self.c2, kernel_size=3, padding=1, stride=2)
        self.conv5 = nn.Conv2d(in_channels=self.c2, out_channels=self.c1, kernel_size=3, padding=1, stride=2)
        self.l1 = nn.Linear(2880, nc)
        self.act = nn.PReLU()
        # self.mp3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # self.l2 = nn.Linear(1024,512)
        # self.multi2 = nn.MultiheadAttention(1024, 4, batch_first=True)
        # self.l3 = nn.Linear(512,256)
        # self.multi3 = nn.MultiheadAttention(512, 4, batch_first=True)
        # self.l4 = nn.Linear(256, nc)
        # self.multi4 = nn.MultiheadAttention(256, 4, batch_first=True)

    # 传入的x是三个layer的输出，分别为下采样了8, 16, 32倍率后的
    # x[i] shape：（batch_size, na * (nc * 5), height, w)
    def forward(self, x):
        x1,x2,x3 = x
        x1 = torch.concatenate((self.conv1(x1),x2), dim=1)
        x1 = self.act(x1)
        x2 = self.act(self.conv2(x2))
        x3 = torch.concatenate((x1,x2,x3), dim=1)
        x3 = self.conv3(x3)
        x3 = self.mp2(x3)
        x3 = self.conv4(x3)
        x3 = self.mp2(x3)
        x3 = self.conv5(x3).reshape(x3.size()[0], -1).contiguous()
        x3 = self.act(self.l1(x3))
        # x3 = self.l1(x3)
        # x3 = self.multi2(x3, x3, x3)[0]
        # x3 = self.act(x3)
        # # x3 = self.l2(x3)
        # x3 = self.mp3(x3)
        #
        # x3 = self.multi3(x3, x3, x3)[0]
        # x3 = self.act(x3)
        # # x3 = self.l3(x3)
        # x3 = self.mp3(x3)
        #
        # x3 = self.multi4(x3, x3, x3)[0]
        # x3 = self.act(x3)
        # x3 = self.l4(x3)
        return x3


class YOLOV5(nn.Module):
    def __init__(self, channels=4, args=params.parse(), device=params.parse().device):
        super(YOLOV5, self).__init__()
        with open(args.weightYaml) as f:
            self.config = yaml.load(f.read(), Loader=yaml.FullLoader)

        # 遍历一轮backbone和head，得到系列的结果
        self.blocks = {}
        # 有些中间值需要保存，这里保存一下
        self.saveIndex = []
        for i, block in enumerate(self.config["backbone"]):
            self.blocks[i] = block
        for i, block in enumerate(self.config["head"]):
            i += len(self.config["backbone"])
            if isinstance(block[0], list):
                self.saveIndex += block[0]
            elif block[0] != -1:
                self.saveIndex.append(block)
            self.blocks[i] = block
        self.saveIndex = list(set(self.saveIndex))
        self.saveIndex.remove(-1)
        self.froms = {}
        self.outChannels = {}
        for idx in range(len(self.blocks.keys())):
            fr = self.blocks[idx][0]
            self.froms[idx] = fr
            num = self.blocks[idx][1]
            bname = self.blocks[idx][2]
            bparams = self.blocks[idx][3]

            if bname not in ['nn.Upsample', 'Concat', 'Detect']:
                bparams[0] *= self.config["width_multiple"]
                bparams[0] = round(bparams[0])
            if num != 1:
                _ = []
                for tij in range(round(num * self.config["depth_multiple"])):
                    _.append(eval(bname)(channels, *bparams).to(device))
                    channels = bparams[0]
                self.blocks[idx] = nn.Sequential(*_)
            else:
                if bname not in ['nn.Upsample', 'Concat', 'Detect']:
                    self.blocks[idx] = eval(bname)(channels, *(bparams)).to(device)
                elif bname == 'Detect':
                    self.detectaParams = self.blocks.pop(idx)
                    continue
                elif bname == 'C3' and len(bparams) > 1:
                    self.blocks[idx] = eval(bname)(bparams[0], shortcut=bparams[1]).to(device)
                else:
                    self.blocks[idx] = eval(bname)(*(bparams)).to(device)
            if bname not in ['nn.Upsample', 'Concat', 'Detect']:
                channels = bparams[0]
            elif bname == 'Concat':
                channels = 0
                for ____ in fr:
                    if ____ == -1:
                        channels += self.outChannels[idx - 1]
                    else:
                        channels += self.outChannels[____]

            self.outChannels[idx] = channels

        self.blocks = {str(i): j for i, j in self.blocks.items()}
        self.blocks = nn.ModuleDict(self.blocks.items())
        _detectChannels = []
        for detectFrom in self.detectaParams[0]:
            _detectChannels.append(self.outChannels[detectFrom])
        self.detect = Detect(channels=_detectChannels)

    def forward(self, x):
        interSave = {}  # 保留一些中间值
        for i in range(len(self.blocks.keys())):
            fr = self.froms[i]
            i = str(i)
            block = self.blocks[i]
            i = int(i)

            # 查看一下要不要多层输入
            if fr == -1:
                x = block(x)
            else:

                __ = []
                for ii in fr:
                    if ii == -1:
                        __.append(x)
                    else:
                        __.append(interSave[ii])
                if i != 25:  # 非Detect层
                    x = block(__)
            if i in self.saveIndex:
                interSave[i] = x
        __ = []
        for ii in self.detectaParams[0]:
            if ii == -1:
                __.append(x)
            else:
                __.append(interSave[ii])
        x = self.detect(__)
        return x

class resnet50(nn.Module):
    def __init__(self):
        super(resnet50, self).__init__()

        self.resnet = torchvision.models.resnet50(weights=True)
        self.resnet.conv1 = nn.Conv2d(1 if args.grey else 4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn = nn.BatchNorm1d(1000)
        self.l = nn.Linear(1000, args.nc)

    def forward(self, x):
        return self.l(self.bn(self.resnet(x)))

if __name__ == "__main__":
    net = YOLOV5()
    net = resnet50()
    print("a")
