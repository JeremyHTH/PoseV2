import torch
import cv2
from PIL import Image
import numpy as np

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp3/weights/best.pt')  # local model
model = torch.hub.load('yolov5','custom', path='yolov5/runs/train/exp16/weights/best.pt', source='local') 

# img = Image.open('human_657.jpg')
img = cv2.imread('human_657.jpg')
img = img[:,:,::-1]
result = model(img)


# result.print()
# result.show()
boxes = result.xyxy[0].tolist()
img = img[:,:,::-1]
cv2.imshow('orginal',img)
for index, box in enumerate(boxes):
    x1,y1,x2,y2 = np.array(box[:4],dtype='int32')
    # data = list(map(lambda:int))
    cropped_img = img[y1:y2,x1:x2]
    cv2.imshow('img{}'.format(index),cropped_img)

cv2.waitKey(0)