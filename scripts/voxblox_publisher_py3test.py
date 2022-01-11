#!/home/zhy/anaconda3/envs/zhy/bin/python
from __future__ import print_function
import sys
import os
import rospy
import cv2
import numpy as np
import PIL.Image

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Vector3, Quaternion, Transform, Point, Pose
from geometry_msgs.msg import TransformStamped
from sensor_msgs.point_cloud2 import PointCloud2, PointField
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils import *


def read_TUM_pose_file(pose_file):
    if not os.path.exists(pose_file):
        print("pose file not exsit")
        return

    with open(pose_file, 'r') as f:
        lines = f.readlines()

    return lines


def read_DTU_files(poses_dir):
    files = os.listdir(poses_dir)
    return files


def read_camera_intrinsic(calib_file_path):
    if not os.path.exists(calib_file_path):
        print("calib file not exist")
        return

    camera_intrinsic = np.identity(3, dtype=np.float32)

    with open(calib_file_path, 'r') as f:
        lines = f.readlines()

    f_info = lines[5].strip().split(' ')
    c_info = lines[6].strip().split(' ')
    fx, fy = float(f_info[0]), float(f_info[1])
    cx, cy = float(c_info[0]), float(c_info[1])

    camera_intrinsic[0, 0], camera_intrinsic[0, 2] = fx, cx
    camera_intrinsic[1, 1], camera_intrinsic[1, 2] = fy, cy

    return camera_intrinsic


# now this is TUM format
if __name__ == '__main__':
    # 0.node init
    rospy.init_node('voxblox_publisher')
    cv_bridge = CvBridge()

    # 1.publishers
    transform_stamped_pub = rospy.Publisher("zhy_transform", TransformStamped, queue_size=100)
    pointcloud2_pub = rospy.Publisher("zhy_pointclouds", PointCloud2, queue_size=100)
    # depth_vis_pub = rospy.Publisher("depth_visual_image", Image, queue_size=100)

    # 2.settings
    data_dir = "/home/zhy/HDD/zhy/datasets/TUM/rgbd_dataset_freiburg1_teddy"
    pose_file_name = "associate.txt"
    calib_file_name = "calib.txt"
    depth_dir = "idx_depth"
    img_dir = "idx_rgb"

    print(sys.version)

    pose_path = os.path.join(data_dir, pose_file_name)
    depths_path = os.path.join(data_dir, depth_dir)
    imgs_path = os.path.join(data_dir, img_dir)
    calib_path = os.path.join(data_dir, calib_file_name)

    img_idx = 0
    frame_id = "vicon"
    tum_depth_scale = 5000.0

    rate = rospy.Rate(5)

    # 3.data loading
    # files = read_DTU_files(data_dir)
    intrinsic = read_camera_intrinsic(calib_path)
    poses = read_TUM_pose_file(pose_path)

    # 4.msgs converting and publishing
    # for idx in range(len(files)):
    #     rospy.loginfo(img_idx)
    #     info_path = os.path.join(data_dir, fiels[idx])

    #     with open(info_path, 'r') as f:
    #         lines = f.readlines()

    while True:
        rospy.loginfo(img_idx)
        
        info = poses[img_idx].strip().split(' ')
        stamp = rospy.Time(float(info[4]))
        img_path = os.path.join(data_dir, info[1])
        depth_path = os.path.join(data_dir, info[3])

        if (not os.path.isfile(img_path)) or (not os.path.isfile(depth_path)):
            break

        # 4-1.read pose info
        tx0, ty0, tz0 = float(info[5]), float(info[6]), float(info[7])
        qx0, qy0, qz0, qw0 = float(info[8]), float(info[9]), float(info[10]), float(info[11])

        # q to R, pretend transformation problem
        Rotation_matrix = R.from_quat([qx0, qy0, qz0, qw0])
        q = Rotation_matrix.as_quat()
        qxd, qyd, qzd, qwd = q[0], q[1], q[2], q[3]

        # 4-2.read img, depth info and transform pixel to 3D points
        img = np.array(PIL.Image.open(img_path))
        depth = np.array(PIL.Image.open(depth_path))
        
        points = pixel_to_3DRGB(depth, img, tum_depth_scale, intrinsic)

        # 4-4.convert to ROS
        transform_stamped_msg = tq_to_TransformStamped(
            Vector3(x=tx0, y=ty0, z=tz0),
            Quaternion(x=qxd, y=qyd, z=qzd, w=qwd),
            child_frame_id="kinect", 
            stamp=stamp, 
            frame_id=frame_id
        )

        pointcloud_msg = xyzrgb_array_to_pointcloud2(points, stamp, "tmp_point")
        # pointcloud_msg = xyz_array_to_pointcloud2(points, stamp, "tmp_point")

        # 4-5.publish
        transform_stamped_pub.publish(transform_stamped_msg)
        pointcloud2_pub.publish(pointcloud_msg)

        rate.sleep()
        img_idx += 1