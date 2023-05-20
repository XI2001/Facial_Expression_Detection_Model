import argparse
import pathlib
import torch

def parse():
    parser = argparse.ArgumentParser(description='training')
    pathlib.Path('../FacialExpression/Train/EmtionalTrainingData/Train/images')
    parser.add_argument('--imageDir', type=str, default=r'../FacialExpression/Train/EmtionalTrainingData/backup/images')
    parser.add_argument('--labelDir', type=str, default=r'../FacialExpression/Train/EmtionalTrainingData/backup/labels')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--imgsize', type=int, default=640)
    parser.add_argument('--batchSize', type=int, default=12)
    parser.add_argument('--weightYaml', type=str, default=r'yolov5x.yaml')
    parser.add_argument('--nc', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--grey', type=bool, default=True)
    parser.add_argument('--detectImageDir', type=str, default=r'J:\Image')
    parser.add_argument('--detectLabelDir', type=str, default=r'J:\FacialDetectOutCome\FacialDetectOutComeTxt')
    parser.add_argument('--detectBatchSize', type=int, default=300)
    args = parser.parse_args()
    return args