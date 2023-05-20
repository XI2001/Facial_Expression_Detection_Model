# Facial_Expression_Detection_Model
By using yolov5x to extract the face and using resnet50 to extract the facial expression. -> Eliminate the influence of background information
- Currently only support one persons' facial detection (you can do some change in dataloader.py to realize this function)
## yolov5 Facial Detection
### Training
- Refer to yolov5-original model https://github.com/ultralytics/yolov5
- this model use yolov-5x.pt as pretrained weight and use private dataset to do the training (98% precision and recall rate in testing dataset)
### Detect
- The example of Facial Detection usage can refer to OverallTraining.py -> change the code with annotation # # 脸部识别，和FacialExpression的检测搭配使用
- The related weight is saved in yolov5-master/run/exp12/weights/best.pt
- use yolov5 detect.py to detect the wanted figure and open the option --save-txt when using it
## Facial Expression Detection Model
### Detect
- Use the detect.py in EmotionDetect
- Change the option in params.py
  - --detectImageDir is the dir of the original detecting data which is also being used in yolov5
  - --detectLabelDir is the dir which yolov5 provide the position of the face, automatically provide the position
### Training
- Use yolov5 input txt label, Each txt has format like \[type, x, y, w, h\]
- Use training.py, change the params.py to automatically train it
- The model automatically pad the figure to 640*640, for larger figure, you have to change the Final output Linear layer of the Model to adapt to the new figure size and outcome
- This model originally use resnet50 as the backbone, for lower overfitting probability, you can use resnet18 or VGG to subsitute the origianl model

## Weight
- yolov5 Facial Detection model weight -> yolov5x.pt 
  - Url：https://pan.baidu.com/s/1SbAkwlU75EKyybO5Xra1bw 
  - password:47ch 
- Facial Expression Detection model weight -> resnet50.pt
  - Url：https://pan.baidu.com/s/1MBM559m5kXAKyW0_AHxdbw 
  - password:bcpb 

## Notation
- Using yolov5 original backbone to extract Facial Expression is hard to convergent during the training
