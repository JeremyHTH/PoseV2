#!/usr/bin/env python3
from __future__ import print_function

import cv2
import time  
import Pose_util.PoseModule as pm
import rospy
import math
import PIL.Image
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from Pose_util.hand_angle_dataset import hand_angle_dataset
from yolov4_tiny.yolo import YOLO

class combined_detection():

    def __init__(self):
        self.cvImage_Subscriber = rospy.Subscriber('/camera/color/image_raw',Image,self.cvImage_Subscriber_Callback)
        self.depthImage_Subscriber = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw',Image,self.depthImage_Subscriber_Callback)
        self.cameraInfo_subscriber = rospy.Subscriber('/camera/color/camera_info',CameraInfo,self.Info_Subscriber_Callback)

        self.human_image_Publisher = rospy.Publisher('/detected_human',Image,queue_size=10)
        self.human_gesture_Publisher = rospy.Publisher('/detected_human_gesture',String,queue_size=10)

        self.color_image = None
        self.depth_image = None
        self.image_length = None
        self.image_width = None
        self.depth = None
        self.bridge = CvBridge()
        self.PIL_img = None
        self.cv_img = None
        self.num_of_people = 0

        self.detector = pm.poseDetector()
        self.angle_data = hand_angle_dataset()
        self.yolo = YOLO()

    def cvImage_Subscriber_Callback(self,data):
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            depth_data = []

        except CvBridgeError as e:
            print(e)
        else:
            try:
                self.PIL_img = PIL.Image.fromarray(cv2.cvtColor(self.color_image,cv2.COLOR_BGR2RGB))
                self.PIL_img,boxs = self.yolo.detect_image(self.PIL_img)
                self.cv_img = cv2.cvtColor(np.array(self.PIL_img), cv2.COLOR_RGB2BGR)
            except:
                cv2.imshow('image',self.color_image)
            else:
                if len(boxs) != 0:
                    for id, box in enumerate(boxs):
                        try:
                            top,left,bottom,right = box[:4]
                            # cropped_img = self.color_image[box[0]:box[2],box[1]:box[3]]
                            centre_y = (top+bottom)//2
                            centre_x = (left+right)//2
                            cv2.circle(self.cv_img,(centre_x,centre_y), 5, (0,100,200),cv2.FILLED)
                            cropped_img = self.depth_image[centre_y-25:centre_y+25,centre_x-25:centre_x+25]
                            avg = np.mean(cropped_img)
                            depth_data.append(np.mean(cropped_img))

                        except: 
                            print('detection{} error'.format(id))

                if len(depth_data) != 0:
                    depth_data = np.array(depth_data)
                    minimum_index = np.argmin(depth_data)
                    minimum_index = 0
                    box = boxs[minimum_index] 
                    top,left,bottom,right = self.angle_data.regulation_box(box,self.image_length,self.image_width)
                    print(minimum_index,top,left,bottom,right)
                    try:
                        cropped_img = self.color_image[top:bottom,left:right]
                        cropped_img = self.detector.findPose(cropped_img,True)
                        lmList = self.detector.findPosition(cropped_img,True)
                        Left_straight, Right_straight, Left_angle, Right_angle, Gesture =self.angle_data.cal_angle(lmList)
                        depth_of_human,cx,cy = self.angle_data.depth_calculation(lmList,self.depth_image,left,top)
                    except:
                        print('pose detection error')
                    cv2.imshow('img_detect'.format(id),cropped_img)
                
                
                cv2.imshow('image',self.cv_img)

        if cv2.waitKey(30) & 0xFF == ord('d'):
            print(self.color_image.shape)
            print(self.depth_image.shape)


    def Info_Subscriber_Callback(self,data):
        self.image_length = int(data.P[2]*2)
        self.image_width = int(data.P[6]*2)
        # rospy.loginfo(" length:%s width:%s",data.s[2],data.P[6])

    def depthImage_Subscriber_Callback(self,data):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
        except CvBridgeError as e:
            print(e)


def main():
    rospy.init_node('PostDetectV3', anonymous=True)
    glo_detection = combined_detection()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    # while not rospy.is_shutdown():
    #     rospy.sleep(100)
    

if __name__== "__main__":
    main()
