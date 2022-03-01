import time 

import cv2
import numpy as np
from PIL import Image
from yolov4_tiny.yolo import YOLO
import Pose_util.PoseModule as pm

def main():
    yolo = YOLO()

    image= Image.open('testing_source/human_657.jpg')
    class_list = _get_classlist('yolov4_tiny/model_data/human_class.txt')
    new_img,data = yolo.detect_image(image)
    detector = pm.poseDetector()
    # new_img.show()
    for x in data:
        img = image.crop((x[1],x[0],x[3],x[2]))
        img = PIL2CV(img)
        img = detector.findPose(img,True)
        lmList = detector.findPosition(img,True)
        img = Image.fromarray(img[:,:,::-1])
        img.show()
    
    cv2.waitKey(0)

def testing():
    yolo = YOLO()

    cv_img = cv2.imread('testing_source/human_657.jpg')
    PIL_img = Image.fromarray(cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB))
    new_img,data = yolo.detect_image(PIL_img)
    PIL_img.show()
    cv_img = cv2.cvtColor(np.array(PIL_img), cv2.COLOR_RGB2BGR)
    cv2.imshow('img',cv_img)
    cv2.waitKey(0)



def _get_classlist(path):
    classlist = []
    with open(path,'r') as f:
        lines = f.readlines()
    
    # for line in lines:
    #     classlist.append(line.strip())
    classlist = list(map(lambda x: x.strip(),lines))

    return classlist


def PIL2CV(img):
    # opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)

    cv_Img = np.array(img)
    return cv_Img[:,:,::-1].copy()

if __name__ == "__main__":
    testing()
