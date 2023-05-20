import torch.nn

import model
import params
import dataloader
from model import YOLOV5, resnet50
from torch.cuda.amp import autocast as autocast
from tqdm import tqdm
import pandas as pd


EmoTrainData, EmoTestData = dataloader.getDataset()
# net = YOLOV5()
net = resnet50()
net = net.to('cuda')
loss = torch.nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
scheduler = torch.optim.lr_scheduler.CyclicLR(optim, base_lr=0.0001, max_lr=0.001, gamma=0.9, step_size_up=10)
log = pd.DataFrame(columns=['序号', '训练loss', '训练acc', '测试loss', '测试acc'])
def calPre(x, label):
    _ = torch.softmax(x,1)
    # print(torch.argmax(x,dim=1))
    return torch.sum(torch.abs(torch.argmax(_,dim=1) - label) < 1e-4) / label.size()[0]

def Mean(a):
    return sum(a) / len(a)

for epoch in range(params.parse().epochs):
    print("epoch{}/{}".format(epoch, params.parse().epochs))
    bar = tqdm(EmoTrainData, total=len(EmoTrainData))
    with autocast():
        precisions = []
        losses = []
        for data in bar:
            x = data[0]
            label = data[1]
            x = net(x)
            l = loss(x, label)
            l.backward()
            optim.step()
            optim.zero_grad()
            p = calPre(x, label)
            torch.cuda.empty_cache()
            bar.set_postfix_str("loss:{}, precision{}".format(l,p))
            scheduler.step()

            precisions.append(p)
            losses.append(l)
        trainloss = Mean(losses).cpu().detach().numpy()
        trainacc =  Mean(precisions).cpu().detach().numpy()
        print("训练集：损失{}；准确率{}".format(trainloss ,trainacc))

        with torch.no_grad():
            net.eval()
            precisions = []
            losses = []
            for data in EmoTestData:
                x = data[0]
                label = data[1]
                x = net(x)
                l = loss(x, label)
                p = calPre(x, label)
                precisions.append(p)
                losses.append(l)
            testloss = Mean(losses).cpu().detach().numpy()
            testacc = Mean(precisions).cpu().detach().numpy()
            print("训练集：损失{}；准确率{}".format(testloss, testacc))
        log.loc[epoch,['序号', '训练loss', '训练acc', '测试loss', '测试acc']] = [epoch, trainloss, trainacc, testloss,testacc]
        if abs(testacc - max(log.loc[:,'测试acc']))<1e-5:
            torch.save(net, 'weights/testAccBest.pt')
            print('TestAcc最佳权重更新')
        if abs(testloss - max(log.loc[:,'测试loss']))<1e-5:
            torch.save(net, 'weights/testLossBest.pt')
            print('TestLoss最佳权重更新')
        if abs(trainacc - max(log.loc[:,'训练acc']))<1e-5:
            torch.save(net, 'weights/trainAccBest.pt')
            print('TrainAcc最佳权重更新')
        if abs(trainloss - max(log.loc[:,'训练loss']))<1e-5:
            torch.save(net, 'weights/trainLossBest.pt')
            print('TrainLoss最佳权重更新')
        net.train()
        log.to_csv('训练结果.csv')



