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
import cv2
from cv_bridge import CvBridge
import os
import message_filters
from datetime import datetime

now = datetime.now()

date = now.strftime("%m_%d_%Y_%H_%M_%S")

if not os.path.exists("data"):
    os.makedirs("data")

os.makedirs(f'data/{date}/images')
os.makedirs(f'data/{date}/odom')

print(f'Successfully created directories for {date}.')

bridge = CvBridge()

imgcount = 0

def callback(image, odom):
    global imgcount
    # Save image data
    image = bridge.imgmsg_to_cv2(image)
    cv2.imwrite(f'data/{date}/images/{imgcount:09}.png', image)

    # Save odom data
    f = open(f'data/{date}/odom/{imgcount:09}.yml', "x")
    f.write(str(odom.pose.pose))
    f.close()

    imgcount+=1


def listener():
    rospy.init_node('pose_recorder', anonymous=True)
    odom_sub = message_filters.Subscriber("odom", Odometry)
    image_sub = message_filters.Subscriber("/camera/image", Image)

    ts = message_filters.ApproximateTimeSynchronizer([image_sub, odom_sub], queue_size=100, slop=0.1)

    ts.registerCallback(callback)

    print("Synchronizer started")

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

print("Starting to listen for data...")

listener()
