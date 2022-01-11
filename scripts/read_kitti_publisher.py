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
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Vector3, Quaternion, Transform, Point, Pose
from geometry_msgs.msg import TransformStamped
from sensor_msgs.point_cloud2 import PointCloud2, PointField
import numpy as np

from utils import *

import os

if __name__ == '__main__':
    rospy.init_node('kitti_publisher')
    cv_bridge = CvBridge()
    transform_stamped_pub = rospy.Publisher("zhy_transform", TransformStamped, queue_size=100)
    pointcloud2_pub = rospy.Publisher("zhy_pointclouds", PointCloud2, queue_size=100)

    #path_name = "/home/zhy/datasets/kitti_odo/sequences/00"
    path_name = "/home/zhy/HDD/zhy/datasets/kitti_sequence_00/00"
    image_index = 0
    rate = rospy.Rate(5)

    poses = []
    poses_file = "/home/zhy/HDD/zhy/datasets/kitti_odo/dataset/poses/00.txt"

    with open(poses_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        info = line.strip().split(' ')
        nowpose = np.eye(4, dtype=np.float32)
        nowpose[0, 0], nowpose[0, 1], nowpose[0, 2], nowpose[0, 3] = float(info[0]), float(info[1]), float(info[2]), float(info[3])
        nowpose[1, 0], nowpose[1, 1], nowpose[1, 2], nowpose[1, 3] = float(info[4]), float(info[5]), float(info[6]), float(info[7])
        nowpose[2, 0], nowpose[2, 1], nowpose[2, 2], nowpose[2, 3] = float(info[8]), float(info[9]), float(info[10]), float(info[11])
        poses.append(nowpose)

    intrinsic = np.array([[718.856, 0.0, 607.1928],
                          [0.0, 718.856, 185.2157],
                          [0.0, 0.0, 1.0]], dtype=np.float32)

    while not rospy.is_shutdown():
        print(image_index)
        left_img_path = path_name + "/image_0/%06d.png"%image_index
        right_img_path = path_name + "/image_1/%06d.png"%image_index
        depth_path = path_name + "/depth_0/%06d.npy" % image_index
        if (not os.path.isfile(left_img_path)) or (not os.path.isfile(right_img_path)):
            break

        left_img = cv2.imread(left_img_path,1)
        right_img = cv2.imread(right_img_path,1)

        cv2.imshow("left", left_img)
        input_key = cv2.waitKey(10)

        depth_np = np.load(depth_path)
        depth_np = 386.1448 / depth_np  #00-02
        mask = depth_np < 10.0
        depth_np = depth_np * mask.astype(np.float32)
        depth_np[:80, :] = 0.0

        # # depth_np = 379.8145 / depth_np  #04-12
        # depth_visual = np.clip(depth_np, 0.5, 60)
        # depth_visual = depth_visual / 60.0 * 255.0
        # depth_visual = depth_visual.astype(np.uint8)
        # col = cv2.applyColorMap(depth_visual, cv2.COLORMAP_RAINBOW)

        # cv2.imshow("depth", col)
        # cv2.waitKey(10)

        Tcw = poses[image_index]

        # Tcw = np.linalg.inv(Tcw)

        rotation = R.from_matrix(Tcw[:3, :3])
        q = rotation.as_quat()
        tx, ty, tz = Tcw[0, 3], Tcw[1, 3], Tcw[2, 3]
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]

        pub_ros_time = rospy.get_rostime()

        transform_stamped_msg = tq_to_TransformStamped(
            Vector3(x=tx, y=ty, z=tz),
            Quaternion(x=qx, y=qy, z=qz, w=qw),
            child_frame_id="kinect", 
            stamp=pub_ros_time, 
            frame_id="vicon"
        )

        points = pixel_to_3DRGB(depth_np, left_img, 1.0, intrinsic)
        pointcloud_msg = xyzrgb_array_to_pointcloud2(points, pub_ros_time, "tmp_point")

        transform_stamped_pub.publish(transform_stamped_msg)
        pointcloud2_pub.publish(pointcloud_msg)

        rate.sleep()
        image_index+=1
        
        if input_key == 27:
            break
