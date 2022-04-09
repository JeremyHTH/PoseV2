#!/usr/bin/env python3
# license removed for brevity
import rospy
import cv2
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image


def talker():
    img = cv2.imread('')
    bridge = CvBridge()
    ros_img = bridge.cv2_to_imgmsg(img)
    pub = rospy.Publisher('/detection_result/image', Image, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():

        pub.publish(img)
        rate.sleep()

def trapezium_centroid(c,a,b):

    return (2*a*c + a**2 + c*b + a*b + b**2)/(3*(a+b))

if __name__ == '__main__':
    # try:
    #     talker()
    # except rospy.ROSInterruptException:
    #     pass

    print(trapezium_centroid(10,20,50))
