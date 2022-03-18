import torch
import cv2
from PIL import Image

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/best.pt')  # local model
model = torch.hub.load('yolov5','custom', path='yolov5/runs/train/exp16/weights/best.pt', source='local') 

img = Image.open('human_1787_jpg.rf.34ccbda621dd0f8fbeeed80ea6b1b5fc.jpg')
result = model(img)
result.print()
result.show()