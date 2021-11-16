#!/usr/bin/env python
from __future__ import print_function
import sys
import os
import rospy
import cv2
import numpy as np
import PIL.Image

from geometry_msgs.msg import Vector3, Quaternion, Transform, Point, Pose
from geometry_msgs.msg import TransformStamped
from sensor_msgs.point_cloud2 import PointCloud2, PointField
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


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


def depth_image_to_point_cloud(rgb, depth, scale, K, pose):
    u = range(0, rgb.shape[1])
    v = range(0, rgb.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    position = np.vstack((X, Y, Z, np.ones(len(X))))
    position = np.dot(pose, position)

    R = np.ravel(rgb[:, :, 0])[valid]
    G = np.ravel(rgb[:, :, 1])[valid]
    B = np.ravel(rgb[:, :, 2])[valid]

    points = np.transpose(np.vstack((position[0:3, :], R, G, B))).tolist()

    return points


def pixel_to_3D(depth, scale, K):
    u = range(0, depth.shape[1])
    v = range(0, depth.shape[0])

    u, v = np.meshgrid(u, v)
    u = u.astype(float)
    v = v.astype(float)

    Z = depth.astype(float) / scale
    X = (u - K[0, 2]) * Z / K[0, 0]
    Y = (v - K[1, 2]) * Z / K[1, 1]

    X = np.ravel(X)
    Y = np.ravel(Y)
    Z = np.ravel(Z)

    valid = Z > 0

    X = X[valid]
    Y = Y[valid]
    Z = Z[valid]

    res = np.transpose(np.vstack((X, Y, Z)))
    
    return res

    # h, w = depth.shape
    # x, y = np.meshgrid(
    #     np.arange(0, h, dtype=np.float32),
    #     np.arange(0, w, dtype=np.float32)
    # )

    # x = x.ravel()
    # y = y.ravel()

    # res = np.stack((x, y, np.ones_like(x)))
    # res = np.matmul(K_inv, res) # (3, h*w)
    # res = np.expand_dims(depth, axis=0).ravel() *res
    # res = res.ravel()
    # # res = res.reshape(3, h, w)
    # # res = np.expand_dims(depth, axis=0) * res
    # # res = res.transpose((1, 2, 0))

    # return res


def read_pose_file(pose_file):
    if not os.path.exists(pose_file):
        print("pose file not exsit")
        return

    with open(pose_file, 'r') as f:
        lines = f.readlines()

    return lines


def tq_to_TransformStamped(t, q, child_frame_id, stamp=None, frame_id=None, seq=None):
    transform = Transform(
        translation=t,
        rotation=q
    )

    msg = TransformStamped()

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq:
        msg.header.seq = seq

    msg.child_frame_id = child_frame_id
    msg.transform = transform

    return msg


def xyz_array_to_pointcloud2(points, stamp=None, frame_id=None, seq=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()

    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if seq: 
        msg.header.seq = seq

    if len(points.shape) == 3:
        msg.height = points.shape[0]
        msg.width = points.shape[1]
    else:
        msg.height = 1
        msg.width = len(points)

    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()

    return msg


# now this is TUM format
if __name__ == '__main__':
    # 0.node init
    rospy.init_node('voxblox_publisher')
    cv_bridge = CvBridge()

    # 1.publishers
    transform_stamped_pub = rospy.Publisher("transform", TransformStamped, queue_size=100)
    pointcloud2_pub = rospy.Publisher("pointcloud", PointCloud2, queue_size=100)
    # depth_vis_pub = rospy.Publisher("depth_visual_image", Image, queue_size=100)

    # 2.settings
    data_dir = "/home/zhy/HDD/zhy/datasets/TUM/rgbd_dataset_freiburg1_teddy"
    pose_file_name = "associate.txt"
    calib_file_name = "calib.txt"
    depth_dir = "idx_depth"
    img_dir = "idx_rgb"

    pose_path = os.path.join(data_dir, pose_file_name)
    depths_path = os.path.join(data_dir, depth_dir)
    imgs_path = os.path.join(data_dir, img_dir)
    calib_path = os.path.join(data_dir, calib_file_name)

    img_idx = 0
    frame_id = "base_link"
    tum_depth_scale = 5000.0

    rate = rospy.Rate(5)

    # 3.data loading
    intrinsic = read_camera_intrinsic(calib_path)
    poses = read_pose_file(pose_path)

    # 4.msgs converting and publishing
    while True:
        a = input()

        if a == 'e':
            break

        print(img_idx)
        img_path = os.path.join(imgs_path, "%04d.png"%img_idx)
        depth_path = os.path.join(depths_path, "%04d.png"%img_idx)

        if (not os.path.isfile(img_path)) or (not os.path.isfile(depth_path)):
            break

        # 4-1.read pose info
        pose_info = poses[img_idx].strip().split(' ')
        stamp = rospy.Time(float(pose_info[4]))
        tx0, ty0, tz0 = float(pose_info[5]), float(pose_info[6]), float(pose_info[7])
        qx0, qy0, qz0, qw0 = float(pose_info[8]), float(pose_info[9]), float(pose_info[10]), float(pose_info[11])

        # 4-2.read img, depth info and transform pixel to 3D points
        img = cv2.imread(img_path, 1)
        depth = np.array(PIL.Image.open(depth_path))
        points = pixel_to_3D(depth, tum_depth_scale, intrinsic)
        # points = depth_image_to_point_cloud()

        # 4-3.depth visualization
        # depth_vis = cv2.applyColorMap(depth, cv2.COLORMAP_RAINBOW)

        # 4-4.convert to ROS
        transform_stamped_msg = tq_to_TransformStamped(
            Vector3(x=tx0, y=ty0, z=tz0),
            Quaternion(x=qx0, y=qy0, z=qz0, w=qw0),
            child_frame_id=0, 
            stamp=stamp, 
            frame_id=frame_id
        )

        pointcloud_msg = xyz_array_to_pointcloud2(points, stamp, frame_id)
        
        # depth_vis_msg = cv_bridge.cv2_to_imgmsg(depth_vis, "bgr8")
        # depth_vis_msg.header.stamp = stamp

        # 4-5.publish
        transform_stamped_pub.publish(transform_stamped_msg)
        pointcloud2_pub.publish(pointcloud_msg)
        # depth_vis_pub.publish(depth_vis_msg)

        rate.sleep()
        img_idx += 1