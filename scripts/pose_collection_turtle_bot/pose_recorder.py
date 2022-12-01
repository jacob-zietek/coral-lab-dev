#!/usr/bin/env python

'''
This file records turtlebot3 camera poses and odom data
to create a NeRF. This file should be run while the robot
is in teleop mode. 

roslaunch turtlebot3_teleop turtlebot3_teleop_key.launch
'''

# https://robotics.stackexchange.com/questions/6652/how-to-get-a-python-node-in-ros-subscribe-to-multiple-topics
# http://wiki.ros.org/rospy/Overview/Publishers%20and%20Subscribers#Subscribing_to_a_topic

import rospy
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import os


if not os.path.exists("data"):
    os.makedirs("data")

if not os.path.exists("data/images"):
    os.makedirs("data/images")

if not os.path.exists("data/odom"):
    os.makedirs("data/odom")


bridge = CvBridge()

imgcount = 0

def record_image(msg):
    image = bridge.imgmsg_to_cv2(msg)
    cv2.imwrite(f'data/images/{imgcount:09}.png', image)
    imgcount+=1



def record_odom(data):
    print(data)
    


def listener():
    rospy.init_node('pose_recorder', anonymous=True)
    rospy.Subscriber("odom", Odometry, record_odom)
    rospy.Subscriber("/camera/image", Image, record_image)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

listener()
