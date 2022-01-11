#!/home/zhy/anaconda3/envs/zhy/bin/python
from __future__ import print_function
import sys
import os

from torch._C import dtype

import rospy
import cv2
import numpy as np
import PIL.Image
import math
import time
import re
import argparse

from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Vector3, Quaternion, Transform, Point, Pose
from geometry_msgs.msg import TransformStamped
from sensor_msgs.point_cloud2 import PointCloud2, PointField
from sensor_msgs.msg import Image

from utils import *
from datasets import find_dataset_def
from torch.utils.data import DataLoader

import matplotlib as mpl
import matplotlib.cm as cm

# cudnn.benchmark = True


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics


class EuRoc_Test():
    def __init__(self, datapath, seq):
        self.seqpath = os.path.join(datapath, seq)
        self.poses = {}
        self.indexs = []
        self.ext = "jpg"

        self.Kc0 = np.array([[458.654, 0.0, 367.215],
                             [0.0, 457.296, 248.375],
                             [0.0, 0.0, 1.0]], dtype=np.float32)

        self.Kc1 = np.array([[457.587, 0.0, 379.999],
                             [0.0, 456.134, 255.238],
                             [0.0, 0.0, 1.0]], dtype=np.float32)

        self.Dc0 = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float32)
        self.Dc1 = np.array([-0.28368365, 0.07451284, -0.00010473, -3.55590700e-05], dtype=np.float32)

        self.Tbc0 = np.array([[0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                              [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                              [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                              [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.Tc0b = np.linalg.inv(self.Tbc0)

        self.T01 = np.array([[0.999997256477881, 0.00231206719242389, 0.00037600810241559, -0.110073808127187],
                             [-0.00231713572328124, 0.999898048506644, 0.0140898358466482, 0.000399121547014141],
                             [-0.000343393120524168, -0.0140906684527146, 0.999900662637728, -0.000853702503358044],
                             [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        self.T10 = np.linalg.inv(self.T01)

        transform_path = os.path.join(datapath, "gt_poses", "{}.txt".format(seq))

    def get_color(self, side, vid, do_flip):
        img_path = os.path.join(self.seqpath, "images", "{:0>8}.jpg".format(vid))

        img = PIL.Image.open(img_path)

        if do_flip:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        img = img.convert('RGB')

        return np.array(img)

    def get_cam(self, vid):
        ref_intrinsics, ref_extrinsics = read_camera_parameters(os.path.join(self.seqpath, 'cams/{:0>8}_cam.txt'.format(vid)))
        return ref_intrinsics, ref_extrinsics

    # def get_pose(self, side, vid):
    #     T_cw = self.poses[vid]
       
    #     if side == "1":
    #         T_cw = self.T01 @ T_cw
    
    #     return T_cw

    def get_intrinsics(self, side):
        if side == "1":
            K = self.Kc1.copy()
            D = self.Dc1.copy()
        else:
            K = self.Kc0.copy()
            D = self.Dc0.copy()

        return K, D

    def get_depth(self, vid):
        depth = read_pfm(os.path.join(self.seqpath, 'depth_est/{:0>8}.pfm'.format(vid)))[0]
        return depth

    def get_mask(self, vid):
        mask = PIL.Image.open(os.path.join(self.seqpath, 'mask/{:0>8}_final.png'.format(vid)))
        mask = np.array(mask).astype(np.float32) / 255.0
        return mask

    def info_obtain(self, side, vid, do_flip):
        # This place should be changed for T_cw be used in the train.py
        img = self.get_color(side, vid, do_flip)
        depth = self.get_depth(vid)
        mask = self.get_mask(vid)
        # T_cw = self.get_pose(side, vid)
        # K, D = self.get_intrinsics(side)

        intrinsic, extrinsic = self.get_cam(vid)

        # if do_flip:
        #     T_cw = self.flip_array @ T_cw @ self.flip_array

        # return img, depth, mask, T_cw, K

        return img, depth, mask, extrinsic, intrinsic


class NodeParams():
    def __init__(self):
        # base env params
        self.dmax = 5.0
        self.dmin = 0.1
        
        # model params
        self.initN = 192
        self.ndepths = "128,32,8"
        self.fea_channels = 8
        self.cr_base_chs = "8,8,8"
        self.depth_inter_r = "4,2,1"
        self.grad_method = "detach"
        self.refine = False
        self.nores = False
        self.isinv = False
        self.share_cr = False
        self.modelpath = "./checkpoints"

        # fram selection params
        self.rfRange = 5
        self.nviews = 5
        self.imgpath = "./imgs"
        self.depthpath = "./depths"
        self.wb = 1.0
        self.wt = 1.0
        self.conf = 0.95
        self.relthetaMax = 10.0
        self.baselinek = 0.05

    def print_params(self):
        d = self.__dict__

        for key in d:
            if "_" not in key:
                print(key, ':', d[key])


if __name__ == '__main__':
    params = NodeParams()
    
    rospy.init_node("depth_pred_node")

    transform_stamped_pub = rospy.Publisher("zhy_transform", TransformStamped, queue_size=100)
    pointcloud2_pub = rospy.Publisher("zhy_pointclouds", PointCloud2, queue_size=100)

    # local debug
    # base env params
    params.dmax = 10.0
    params.dmin = 0.1

    # model settings
    params.initN = 192
    params.ndepths = "48,32,8"
    params.fea_channels = 8
    params.cr_base_chs = "8,8,8"
    params.depth_inter_r = "4,2,1"
    params.grad_method = "detach"
    params.refine = False
    params.share_cr = False

    params.modelpath = "/home/zhy/catkin_ws/src/data_publisher/scripts/checkpoint/model_000015_gt.ckpt"

    # frame selection settings
    params.rfRange = 1
    params.nviews = 3
    params.imgpath = "/home/zhy/codes/self-sup-MVS-dev/outputs/"
    params.depthpath = "./depths"
    params.wb = 1.0
    params.wt = 1.0
    params.conf = 0.9
    params.relthetaMax = 10
    params.baselinek = 0.05

    params.print_params()

    euroc_test = EuRoc_Test(params.imgpath, "V101")

    nowIdx = 0
    rate = rospy.Rate(0.5)

    vids = ["594", "597", "598", "599", "601", "602", "603", "604", "605", "606",
            "607", "608", "609", "610", "611", "612", "615", "616", "619", "620",
            "650", "653", "663", "664", "670", "675", "680", "686", "693", "698",
            "707", "740", "749", "759", "766", "774", "781", "786", "792", "802",
            "822", "854", "862", "878"]

    while not rospy.is_shutdown() and nowIdx < 3580:
        idx = vids[nowIdx]

        img, depth, mask, Tcw, intrinsic = euroc_test.info_obtain("0", idx, False)
        depth_final = depth * mask

        Twc = np.linalg.inv(Tcw)

        rotation = R.from_matrix(Twc[:3, :3])
        q = rotation.as_quat()
        tx, ty, tz = Twc[0, 3], Twc[1, 3], Twc[2, 3]
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]

        pub_ros_time = rospy.get_rostime()

        transform_stamped_msg = tq_to_TransformStamped(
            Vector3(x=tx, y=ty, z=tz),
            Quaternion(x=qx, y=qy, z=qz, w=qw),
            child_frame_id="kinect", 
            stamp=pub_ros_time, 
            frame_id="vicon"
        )

        points = pixel_to_3DRGB(depth_final, img, 1.0, intrinsic)
        pointcloud_msg = xyzrgb_array_to_pointcloud2(points, pub_ros_time, "tmp_point")

        transform_stamped_pub.publish(transform_stamped_msg)
        pointcloud2_pub.publish(pointcloud_msg)
        
        rate.sleep()
        nowIdx += 1

# rospy.spin()

# [785, 1485]