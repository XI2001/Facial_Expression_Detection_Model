import os
import pathlib
from pathlib import Path

# 执行第一个训练，也就是不包含book，不包含脸部的状况的
# val.py 的287 verbose删除了注释，一定会打印每个类别的准确度
# dataloaders.py打开了albumentations 695行
# --resume是断点训练

os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


# TrainPath = "../TrainData/TrainData.yaml"
# hypPath = "data/hyps/hyp.scratch-low.yaml"
# projectPath = "../Project/NoBookNoReliable/train/"
# os.system(
#     "python train.py --weights {} --data {} --hyp {} --epochs {} --batch-size {} --device 0 --img 640 --workers 1 --cos-lr --rect --cos-lr".format(
#         "weights/yolov5x.pt",
#         TrainPath,
#         hypPath,
#         2000,
#         24,
#     )
# )

#
# TrainPath = pathlib.Path("../FacialRecognition/FacialRecognition.yaml")
# hypPath = "data/hyps/hyp.scratch-high.yaml"
# projectPath = "../Project/NoBookNoReliable/train/"
# os.system(
#     "python train.py --weights {} --data {} --hyp {} --epochs {} --batch-size {} --device 0 --img 640 --workers 1 --cos-lr".format(
#         "weights/yolov5x.pt",
#         TrainPath,
#         hypPath,
#         2000,
#         16,
#     )
# )

# # 表情识别
# TrainPath = pathlib.Path(r"E:\BaiduSyncdisk\Scut\computer\UnstructuredDataAnalysis\ObjectDetect\FacialExpression\Train\EmtionalTrainingData\EmotionalTrain.yaml")
# hypPath = "data/hyps/hyp.scratch-med.yaml"
# os.system(
#     "python train.py --weights {} --data {} --hyp {} --epochs {} --batch-size {} --device 0 --img 640 --workers 1 --cos-lr".format(
#         "weights/yolov5x.pt",
#         TrainPath,
#         hypPath,
#         2000,
#         24,
#     )
# )

# TrainPath = pathlib.Path("../ReliableRecognition/ReliableRecognition.yaml")
# hypPath = "data/hyps/hyp.scratch-low.yaml"
# projectPath = "../Project/NoBookNoReliable/train/"
# os.system(
#     "python train.py --weights {} --data {} --hyp {} --epochs {} --batch-size {} --device 0 --img 640 --workers 1 --cos-lr --rect --cos-lr".format(
#         "weights/yolov5x.pt",
#         TrainPath,
#         hypPath,
#         2000,
#         24,
#     )
# )

# # 脸部识别，和FacialExpression的检测搭配使用
# projectPath = r"J:\FacialDetectOutCome"
# weightPath = r"E:\BaiduSyncdisk\Scut\computer\UnstructuredDataAnalysis\ObjectDetect\yolov5-master\runs\train\exp12\weights\best.pt"
# sourcePath = r"J:\Image"
# os.system(
#     "python detect.py --device 0 --classes 0 --weight {} --project {} --source {} --save-txt".format(weightPath, projectPath, sourcePath)
# )


