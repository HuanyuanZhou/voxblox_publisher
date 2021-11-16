#! /usr/bin/env python
import sys
import os
import rospy
import cv2
import csv
import matplotlib as mpl
import matplotlib.cm as cm
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image, Imu
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

import os

def readImageNames(namePath):
    image_names = os.listdir(namePath)
    # image_names.sort()
    image_names.sort(key=setInt)
    return image_names

def setInt(elem):
    return int(elem.strip().split('.')[0])

def setFirst(elem):
    return elem[0]

def readImu(imuPath):
    imu_data = []
    with open(imuPath, 'r') as f:
        reader = csv.reader(f)
        for line in f:
            if line[0] == '#':
                continue
            info = line.strip('\n').strip('\r').split(',')
            tmp = []
            for i in range(0, len(info)):
                tmp.append(info[i])
            imu_data.append(tmp)
    imu_data.sort(key=setFirst)
    return imu_data

def readTimeStamps(timeStampsPath):
    f = open(timeStampsPath, 'r')
    timestamps = f.readlines()
    f.close()
    return timestamps

if __name__ == '__main__':
    rospy.init_node('EuRoC_publisher')
    cv_bridge = CvBridge()

    # image publisher
    left_pub = rospy.Publisher("left_image", Image, queue_size=100)

    # imu publisher
    imu_pub = rospy.Publisher("imu", Imu, queue_size=5000)
    
    # depth publisher
    depth_pub = rospy.Publisher("depth_image", Image, queue_size=100)

    # depth visual publisher
    depth_visual_pub = rospy.Publisher("depth_visual_image", Image, queue_size=100)

    # map save publisher
    map_save_pub = rospy.Publisher("map_saver", String, queue_size=100)

    # data load path
    # path_name = "/home/zhy/HDD/zhy/datasets/EuRoc/"
    # path_imu = path_name + "MH01/mav0/imu0/data.csv"
    # path_cam = path_name + "MH01/mav0/cam0/data"
    # path_disp = path_name + "MH01/mav0/depth/npy"
    # path_timestamps = path_name + "TimeStamps/MH01.txt"
    path_name = "/home/zhy/HDD/zhy/datasets/EuRoc/"
    path_imu = path_name + "dt_MH01/imu0/data.csv"
    path_cam = path_name + "dt_MH01/cam0/"
    path_disp = path_name + "dt_MH01/depth0/npy"
    path_timestamps = path_name + "TimeStamps/MH01.txt"
    image_index = 0
    imu_index = 0
    rate = rospy.Rate(15)
    cv2.namedWindow("left")
    cv2.moveWindow("left", 100, 100)
    cv2.namedWindow("depth")
    cv2.moveWindow("depth", 100, 100)

    # load data informations
    image_names = readImageNames(path_cam)  # can not use idx for sort reason
    disp_names = readImageNames(path_disp)  # can not use idx for sort reason
    imu_data = readImu(path_imu)
    timestamps = readTimeStamps(path_timestamps)

    while image_index+1 < len(image_names):
        # load left image and its next, next for imu criterion
        left_img_path = path_cam + '/' + image_names[image_index]
        disp_path = path_disp + '/' + disp_names[image_index]
        timestamp = timestamps[image_index].strip()
        next_timestamp = timestamps[image_index+1].strip()

        # assert if image and disp match
        assert image_names[image_index].strip().split('.')[0] == disp_names[image_index].strip().split('.')[0], "image can not match depth!"

        if not os.path.isfile(left_img_path):
            break

        # load and show color image
        left_img = cv2.imread(left_img_path, 0)
        cv2.imshow("left", left_img)
        input_key = cv2.waitKey(1)

        ## load and show depth image
        disp_np = np.load(disp_path)[0]  # [1, h, w]
        depth_np = 1 / disp_np
        depth_np = depth_np.transpose((1, 2, 0))

        vmax = np.percentile(disp_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        col = (mapper.to_rgba(np.squeeze(disp_np))[:, :, :3] * 255).astype(np.uint8)

        col = col[:, :, [2, 1, 0]]

        cv2.imshow("depth", col)
        cv2.waitKey(1)

        stamp_secs = int(timestamp[:10])
        stamp_nsecs = int(timestamp[10:])

        # convert image to ros foramt and publish it with its timestamp
        left_msg = cv_bridge.cv2_to_imgmsg(left_img, "mono8")
        left_msg.header.stamp.secs = stamp_secs
        left_msg.header.stamp.nsecs = stamp_nsecs
        left_pub.publish(left_msg)

        depth_msg = cv_bridge.cv2_to_imgmsg(depth_np, "32FC1")
        depth_msg.header.stamp.secs = stamp_secs
        depth_msg.header.stamp.nsecs = stamp_nsecs
        depth_pub.publish(depth_msg)

        depth_visual_msg = cv_bridge.cv2_to_imgmsg(col, "bgr8")
        depth_visual_msg.header.stamp.secs = stamp_secs
        depth_visual_msg.header.stamp.nsecs = stamp_nsecs
        depth_visual_pub.publish(depth_visual_msg)

        # get imu data and publish
        next_stamp_secs = int(next_timestamp[:10])
        next_stamp_nsecs = int(next_timestamp[10:])
        if imu_index < len(imu_data):
            while (int(imu_data[imu_index][0][:10]) < next_stamp_secs) or ((int(imu_data[imu_index][0][:10]) == next_stamp_secs) and (int(imu_data[imu_index][0][10:]) < next_stamp_nsecs)):
                imu_out = Imu()
                imu_out.header.stamp.secs = int(imu_data[imu_index][0][:10])
                imu_out.header.stamp.nsecs = int(imu_data[imu_index][0][10:])
                imu_out.linear_acceleration.x = float(imu_data[imu_index][4])
                imu_out.linear_acceleration.y = float(imu_data[imu_index][5])
                imu_out.linear_acceleration.z = float(imu_data[imu_index][6])
                imu_out.angular_velocity.x = float(imu_data[imu_index][1])
                imu_out.angular_velocity.y = float(imu_data[imu_index][2])
                imu_out.angular_velocity.z = float(imu_data[imu_index][3])
                imu_pub.publish(imu_out)
                imu_index += 1

        rate.sleep()
        image_index+=1

        if input_key == 27:
            break
        
    # left_img_path = path_cam + "/%s.jpg" %image_names[image_index]
    timestamp = timestamps[image_index].strip()

    left_img_path = path_cam + '/' + image_names[image_index]
    left_img = cv2.imread(left_img_path,0)

    cv2.imshow("left", left_img)
    cv2.waitKey(1)

    left_msg = cv_bridge.cv2_to_imgmsg(left_img, "mono8")
    left_msg.header.stamp.secs = int(timestamp[:10])
    left_msg.header.stamp.nsecs = int(timestamp[10:])
    left_pub.publish(left_msg)
