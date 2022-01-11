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
import torch
import torch.nn as nn
import time
import argparse
import torch.backends.cudnn as cudnn

from kitti_models import *
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import Vector3, Quaternion, Transform, Point, Pose
from geometry_msgs.msg import TransformStamped
from sensor_msgs.point_cloud2 import PointCloud2, PointField
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from utils import *
from datasets import find_dataset_def
from torch.utils.data import DataLoader

import matplotlib as mpl
import matplotlib.cm as cm

cudnn.benchmark = True


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

        with open(transform_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            info = line.strip().split(' ')
            # idx = str(info[0].split('.')[0])
            idx = info[0]
            tx, ty, tz = float(info[1]), float(info[2]), float(info[3])
            qx, qy, qz, qw = float(info[4]), float(info[5]), float(info[6]), float(info[7])

            rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

            Twc = np.eye(4, dtype=np.float32)
            Twc[:3, :3] = rot_matrix
            Twc[0, 3], Twc[1, 3], Twc[2, 3] = tx, ty, tz

            # Tcw = self.Tc0b @ np.linalg.inv(Twb)
            # self.poses[idx] = Tcw
            self.poses[idx] = np.linalg.inv(Twc)
            self.indexs.append(idx)

    def get_color(self, side, vid, do_flip):
        # img_path = os.path.join(self.seqpath, "mav0", "cam{}".format(side), "data", "{}.{}".format(vid, self.ext))
        img_path = os.path.join(self.seqpath, "cam{}".format(side), "data", "{}.{}".format(vid, self.ext))

        img = PIL.Image.open(img_path)

        if do_flip:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        img = img.convert('RGB')

        return np.array(img, dtype=np.float32) / 255.

    def get_pose(self, side, vid):
        T_cw = self.poses[vid]
       
        if side == "1":
            T_cw = self.T01 @ T_cw
    
        return T_cw

    def get_intrinsics(self, side):
        # calib_path = os.path.join(self.seqpath, "mav0", "cam{}".format(side), "sensor.yaml")

        # with open(calib_path, 'r') as f:
        #     lines = f.readlines()
        #     dict_name = "intrinsics"

        #     K = np.zeros((3, 3), dtype=np.float32)
        #     K[2, 2] = 1

        #     D = np.zeros(4, dtype=np.float32)

        #     info = lines[17][13:47]
        #     info = np.fromstring(info, dtype=np.float32, sep=',')
        #     fx, fy, cx, cy = info[0], info[1], info[2], info[3]

        #     K[0, 0], K[0, 2] = fx, cx
        #     K[1, 1], K[1, 2] = fy, cy

        #     D = np.fromstring(lines[19][26:77], dtype=np.float32, sep=',')

        if side == "1":
            K = self.Kc1.copy()
            D = self.Dc1.copy()
        else:
            K = self.Kc0.copy()
            D = self.Dc0.copy()

        return K, D

    def img_preprocess(self, img):
        h, w = img.shape[0], img.shape[1]
        wt, ht = (w // 32) * 32, (h // 32) * 32
        
        ws, hs = (w-wt) // 2, (h-ht) // 2
        img_crop = img[hs: hs+ht, ws:ws+wt]

        return img_crop

    def info_obtain(self, imgs, projs, side, vid, do_flip):
        # This place should be changed for T_cw be used in the train.py
        img = self.get_color(side, vid, do_flip)
        T_cw = self.get_pose(side, vid)
        K, D = self.get_intrinsics(side)

        # img_vis0 = PIL.Image.fromarray((img * 255.0).astype(np.uint8))
        # img_vis0.show()
        
        # img = cv2.undistort(img, K, D, None, None)

        # img_vis1 = PIL.Image.fromarray((img * 255.0).astype(np.uint8))
        # img_vis1.show()

        h, w, _ = img.shape
        img = self.img_preprocess(img)
        ht, wt, _ = img.shape
        
        K[0, 2] = K[0, 2] - (w - wt) // 2
        K[1, 2] = K[1, 2] - (h - ht) // 2

        if do_flip:
            T_cw = self.flip_array @ T_cw @ self.flip_array

        proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
        proj_mat[0, :4, :4] = T_cw
        proj_mat[1, :3, :3] = K

        projs.append(proj_mat)
        imgs.append(img)


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

    # # base env params
    # params.dmax = rospy.get_param('~dmax')
    # params.dmin = rospy.get_param('~dmin')
    
    # # model settings
    # params.initN = rospy.get_param('~initN')
    # params.ndepths = rospy.get_param('~ndepths')
    # params.fea_channels = rospy.get_param('~fea_channels')
    # params.cr_base_chs = rospy.get_param('~cr_base_chs')
    # params.depth_inter_r = rospy.get_param('~depth_inter_r')
    # params.grad_method = rospy.get_param('~grad_method')
    # params.refine = rospy.get_param('~refine')
    # params.nores = rospy.get_param('~nores')
    # params.isinv = rospy.get_param('~isinv')
    # params.share_cr = rospy.get_param('~share_cr')

    # params.modelpath = rospy.get_param('~modelpath')

    # # frame selection settings
    # params.rfRange = rospy.get_param('~rfRange')
    # params.nviews = rospy.get_param('~nviews')
    # params.imgpath = rospy.get_param('~imgpath')
    # params.depthpath = rospy.get_param("~depthpath")
    # params.wb = rospy.get_param('~wb')
    # params.wt = rospy.get_param('~wt')
    # params.conf = rospy.get_param('~conf')
    # params.relthetaMax = rospy.get_param('~relthetaMax')
    # params.baselinek = rospy.get_param('~baselinek')

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

    params.modelpath = "/home/zhy/catkin_ws/src/data_publisher/scripts/checkpoint/kitti_model_000015.ckpt"

    # frame selection settings
    params.rfRange = 1
    params.nviews = 3
    params.imgpath = "/home/zhy/HDD/zhy/datasets/EuRoc/undistorted"
    params.depthpath = "./depths"
    params.wb = 1.0
    params.wt = 1.0
    params.conf = 0.9
    params.relthetaMax = 10
    params.baselinek = 0.05

    params.print_params()

    # model initialize
    model = CascadeMVSNet(
        refine=params.refine, 
        ndepths=[int(nd) for nd in params.ndepths.split(",") if nd],
        depth_interals_ratio=[float(d_i) for d_i in params.depth_inter_r.split(",") if d_i],
        share_cr=params.share_cr,
        cr_base_chs=[int(ch) for ch in params.cr_base_chs.split(",") if ch],
        grad_method=params.grad_method,
    )

    # model = CascadeMVSNet(
    #     refine=params.refine, 
    #     ndepths=[int(nd) for nd in params.ndepths.split(",") if nd],
    #     depth_interals_ratio=[float(d_i) for d_i in params.depth_inter_r.split(",") if d_i],
    #     share_cr=params.share_cr,
    #     cr_base_chs=[int(ch) for ch in params.cr_base_chs.split(",") if ch],
    #     grad_method=params.grad_method,
    #     fea_channels=params.fea_channels,
    #     nores = params.nores,
    #     isinv = params.isinv
    # )

    # load checkpoint file specified by params.loadckpt
    print("loading model {}".format(params.modelpath))
    state_dict = torch.load(params.modelpath, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    euroc_test = EuRoc_Test(params.imgpath, "V101")

    nowIdx = 300
    frame_id = "vicon"
    rate = rospy.Rate(5)

    with torch.no_grad():
        while not rospy.is_shutdown() and nowIdx < 400:
            # if nowIdx % 5 != 0:
            #     nowIdx += 1
            #     continue

            imgs = []
            projs = []
            # vids = [
            #     "616",
            #     "603",
            #     "609",
            #     "634",
            #     "651"
            # ]
            
            vids = [
                euroc_test.indexs[nowIdx], 
                euroc_test.indexs[nowIdx-2], 
                euroc_test.indexs[nowIdx-4],
                euroc_test.indexs[nowIdx+2], 
                euroc_test.indexs[nowIdx+4]
            ]

            depth_values = None

            for vid in vids:
                euroc_test.info_obtain(imgs, projs, "0", vid, False)

                if vid == vids[0]:
                    # # load stereo image information
                    euroc_test.info_obtain(imgs, projs, "1", vid, False)

                    # load depth range information
                    depth_min, depth_max = params.dmin, params.dmax
                    depth_interval = (depth_max - depth_min) / params.initN
                    depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

            imgs = np.stack(imgs).transpose([0, 3, 1, 2])
            proj_matrices = np.stack(projs)

            stage1_pjmats = proj_matrices.copy()
            stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4
            stage2_pjmats = proj_matrices.copy()
            stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2
            stage3_pjmats = proj_matrices.copy()

            proj_matrices_ms = {
                "stage1": torch.from_numpy(stage1_pjmats).unsqueeze(0),
                "stage2": torch.from_numpy(stage2_pjmats).unsqueeze(0),
                "stage3": torch.from_numpy(stage3_pjmats).unsqueeze(0)
            }

            sample = {
                "imgs": torch.from_numpy(imgs).unsqueeze(0),
                "proj_matrices": proj_matrices_ms,
                "depth_values":torch.from_numpy(depth_values).unsqueeze(0)
            }

            sample_cuda = tocuda(sample)
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            del sample_cuda

            nowImage = sample["imgs"][0][0].numpy()
            nowImage = np.transpose(nowImage, (1, 2, 0))
            nowImage = (nowImage * 255.0).astype(np.uint8)
            nowImage_vis = PIL.Image.fromarray(nowImage)
            # nowImage_vis.show()
            #nowImage_vis.save(os.path.join("/home/zhy/HDD/zhy/tmp/img", euroc_test.indexs[nowIdx]+".png"))

            depth_est = outputs["depth"][0]
            # vmin, vmax = params.dmin, params.dmax
            vmin, vmax = depth_est.min(), np.percentile(depth_est, 95)
            normalizer = mpl.colors.Normalize(vmin=-vmax, vmax=-vmin)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            depth_est_vis = (mapper.to_rgba(-depth_est.squeeze())[:, :, :3] * 255).astype(np.uint8)
            depth_est_vis = PIL.Image.fromarray(depth_est_vis)
            # depth_est_vis.show()
            depth_est_vis.save(os.path.join("/home/zhy/HDD/zhy/tmp/depth", euroc_test.indexs[nowIdx]+".png"))

            photometirc_confidence = outputs["photometric_confidence"][0]
            photo_mask = photometirc_confidence > 0.9
            # depth_est = depth_est * photo_mask

            Tcw = sample["proj_matrices"]["stage3"][0,0,0] # Tcw
            pose = Tcw
            # pose = np.linalg.inv(Tcw)
            rotation = R.from_matrix(pose[:3, :3])
            q = rotation.as_quat()
            tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
            qx, qy, qz, qw = q[0], q[1], q[2], q[3]

            timeStamp = rospy.Time.now()

            transform_stamped_msg = tq_to_TransformStamped(
                Vector3(x=tx, y=ty, z=tz),
                Quaternion(x=qx, y=qy, z=qz, w=qw),
                child_frame_id="kinect", 
                stamp=timeStamp, 
                frame_id=frame_id
            )

            intrinsic = sample["proj_matrices"]["stage3"][0,0,1][:3, :3].numpy()

            points = pixel_to_3DRGB(depth_est, nowImage, 1.0, intrinsic)
            pointcloud_msg = xyzrgb_array_to_pointcloud2(points, timeStamp, "tmp_point")

            transform_stamped_pub.publish(transform_stamped_msg)
            pointcloud2_pub.publish(pointcloud_msg)
            rate.sleep()
            nowIdx += 1
            
            # vmin, vmax = depth_est.min(), np.percentile(depth_est, 95)
            # normalizer = mpl.colors.Normalize(vmin=-vmax, vmax=-vmin)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # depth_est_vis = (mapper.to_rgba(-depth_est.squeeze())[:, :, :3] * 255).astype(np.uint8)
            # depth_est_vis = PIL.Image.fromarray(depth_est_vis)
            # depth_est_vis.show()
            # depth_est_vis.save(os.path.join(outdir, ref_name+".png"))

    # rospy.spin()

    # [785, 1485]