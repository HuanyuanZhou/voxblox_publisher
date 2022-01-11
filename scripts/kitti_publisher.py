#!/usr/bin/env python
from __future__ import print_function
import sys
import os

from torch._C import dtype
import rospy
import cv2
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import os

if __name__ == '__main__':
    rospy.init_node('kitti_publisher')
    cv_bridge = CvBridge()
    # left_pub = rospy.Publisher("left_image", Image, queue_size=100)
    left_pub = rospy.Publisher("left_image", Image, queue_size=100)
    right_pub = rospy.Publisher("right_image", Image, queue_size=100)
    depth_pub = rospy.Publisher("depth_image", Image, queue_size=100)
    depth_visual_pub = rospy.Publisher("depth_visual_image", Image, queue_size=100)

    #path_name = "/home/zhy/datasets/kitti_odo/sequences/00"
    path_name = "/home/zhy/HDD/zhy/datasets/kitti_odo/00"
    image_index = 0
    rate = rospy.Rate(5)
    cv2.namedWindow("left")
    cv2.moveWindow("left", 100, 100)

    poses = []
    poses_file = "/home/zhy/HDD/zhy/datasets/kitti_odo/dataset/poses/00.txt"

    with open(poses_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        info = line.strip().split(' ')
        nowpose = np.eye(4, dtype=np.float32)
        nowpose[0, 0], nowpose[0, 1], nowpose[0, 2], nowpose[0, 3] = float(info[0]), float(info[1]), float(info[2]), float(info[3])
        nowpose[1, 0], nowpose[1, 1], nowpose[1, 2], nowpose[1, 3] = float(info[4]), float(info[5]), float(info[6]), float(info[7])
        nowpose[1, 0], nowpose[2, 1], nowpose[2, 2], nowpose[2, 3] = float(info[8]), float(info[9]), float(info[10]), float(info[11])
        poses.append(nowpose)

    while True:
        print(image_index)
        left_img_path = path_name + "/image_0/%06d.png"%image_index
        right_img_path = path_name + "/image_1/%06d.png"%image_index
        depth_path = path_name + "/depth_0/%06d.npy" % image_index
        if (not os.path.isfile(left_img_path)) or (not os.path.isfile(right_img_path)):
            break

        left_img = cv2.imread(left_img_path,0)
        right_img = cv2.imread(right_img_path,0)

        cv2.imshow("left", left_img)
        input_key = cv2.waitKey(10)

        depth_np = np.load(depth_path)
        depth_np = 386.1448 / depth_np  #00-02
        # depth_np = 379.8145 / depth_np  #04-12
        depth_visual = np.clip(depth_np, 0.5, 60)
        depth_visual = depth_visual / 60.0 * 255.0
        depth_visual = depth_visual.astype(np.uint8)
        col = cv2.applyColorMap(depth_visual, cv2.COLORMAP_RAINBOW)

        # cv2.imshow("depth", col)
        # cv2.waitKey(10)

        pub_ros_time = rospy.get_rostime()
        left_msg = cv_bridge.cv2_to_imgmsg(left_img, "mono8")
        left_msg.header.stamp = pub_ros_time
        left_pub.publish(left_msg)

        right_msg = cv_bridge.cv2_to_imgmsg(right_img, "mono8")
        right_msg.header.stamp = pub_ros_time
        right_pub.publish(right_msg)

        depth_msg = cv_bridge.cv2_to_imgmsg(depth_np, "32FC1")
        depth_msg.header.stamp = pub_ros_time
        depth_pub.publish(depth_msg)

        depth_visual_msg = cv_bridge.cv2_to_imgmsg(col, "bgr8")
        depth_visual_msg.header.stamp = pub_ros_time
        depth_visual_pub.publish(depth_visual_msg)

        rate.sleep()
        image_index+=1
        
        if input_key == 27:
            break
