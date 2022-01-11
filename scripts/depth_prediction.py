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

from models import *
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


class TransformGrabber():
    def __init__(
        self, 
        depthModel, calib, distort, 
        dmin, dmax, initN,
        imgPath, depthPath, RFRange, Views,
        wb, wt, relthetaMax, conf,
        baselinek
    ):
        self.mImgPath = imgPath
        self.mDepthPath = depthPath
        self.mRFRange = RFRange
        self.mViews = Views
        
        self.vcalib = calib
        self.vdistort = distort

        self.wb = wb
        self.wt = wt
        self.conf = conf
        self.relthetaMax = relthetaMax
        self.baselineMin = 0.8
        self.dvalues = np.arange(dmin, dmax, (dmax-dmin) / (initN-1), dtype=np.float32)

        self.vTransformBuffer = []
        self.DepthModel = depthModel
        self.PCL2Publisher = rospy.Publisher("zhy_pointclouds", PointCloud2, queue_size=100)

    def calRFScore(self, nPose, rfPose):
        n_tx, n_ty, n_tz = nPose[0], nPose[1], nPose[2]
        n_qx, n_qy, n_qz, n_qw = nPose[3], nPose[4], nPose[5], nPose[6]
        nR = (R.from_quat([n_qx, n_qy, n_qz, n_qw])).as_matrix()

        rf_tx, rf_ty, rf_tz = rfPose[0], rfPose[1], rfPose[2]
        rf_qx, rf_qy, rf_qz, rf_qw = nPose[3], nPose[4], nPose[5], nPose[6]
        rfR = (R.from_quat([rf_qx, rf_qy, rf_qz, rf_qw])).as_matrix()

        Tnw = np.eye(4, dtype=np.float32)
        Tnw[:3, :3] = nR
        Tnw[0, 3], Tnw[1, 3], Tnw[2, 3] = n_tx, n_ty, n_tz

        Trfw = np.eye(4, dtype=np.float32)
        Trfw[:3, :3] = rfR
        Trfw[0, 3], Trfw[1, 3], Trfw[2, 3] = rf_tx, rf_ty, rf_tz

        T_rel = Trfw @ np.linalg.inv(Tnw)

        related_baseline = math.sqrt(T_rel[0, 3]**2 + T_rel[1, 3]**2 + T_rel[2, 3]**2)
        relR = R.from_matrix(T_rel[:3, :3])
        relRotVec = relR.as_rotvec()
        related_theta = math.sqrt(relRotVec[0]**2 + relRotVec[1]**2 + relRotVec[2]**2)
        
        score = 0.0

        # translation score
        score += math.exp(-self.wb*((related_baseline - self.baselineMin)**2))

        # # rotation score
        # if related_theta > self.relthetaMax:
        #     score += 1.0
        # else:
        #     score += math.exp(-self.wt*(self.relthetaMax - related_theta))

        return score

    def selectRF(self, nIdx):
        nPose = self.vTransformBuffer[nIdx][1]
        retIdxArray = [nIdx]
        cmpArray = []

        for i in range(-self.mRFRange, self.mRFRange+1):
            if i == 0:
                continue
            tmpPose = self.vTransformBuffer[nIdx+i][1]
            tmpScore = self.calRFScore(nPose, tmpPose)

            cmpArray.append((tmpScore, nIdx+i))

        cmpArray.sort(reverse=True)

        for i in range (self.mViews-1):
            retIdxArray.append(cmpArray[i][1])

        return retIdxArray

    def dataPrepare(self, idxArray):
        # data process
        imgs = []
        proj_matrices = []

        for i in range(len(idxArray)):
            idx = idxArray[i]
            
            # temp test for image
            # stamp = str(int(self.vTransformBuffer[idx][0]))
            stamp = str((self.vTransformBuffer[idx][0]))
            pose = self.vTransformBuffer[idx][1]

            img = PIL.Image.open(os.path.join(self.mImgPath, stamp+".jpg"))
            img = np.array(img, dtype=np.float32) / 255.

            img = cv2.undistort(img, self.vcalib, self.vdistort, None)
            # img_undist_vis = PIL.Image.fromarray((img * 255).astype(np.uint8))
            # img_undist_vis.show()

            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)

            # img crop form topleft, do not need to change intrinsic
            h, w, _ = img.shape

            wresize = w // 32 * 32
            hresize = h // 32 * 32
            img = img[:hresize, :wresize]

            tx, ty, tz = pose[0], pose[1], pose[2]
            qx, qy, qz, qw = pose[3], pose[4], pose[5], pose[6]
            rot_matrix = R.from_quat([qx, qy, qz, qw]).as_matrix()

            Tcw = np.eye(4, dtype=np.float32)
            Tcw[:3, :3] = rot_matrix
            Tcw[0, 3], Tcw[1, 3], Tcw[2, 3] = tx, ty, tz

            # tmp for TUM test
            # Tcw = np.linalg.inv(Tcw)

            intrinsic = self.vcalib.copy()
            intrinsic[:2, :] /= 4.0

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = Tcw
            proj_mat[1, :3, :3] = intrinsic
            proj_matrices.append(proj_mat)

            imgs.append(img)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        stage2_proj = proj_matrices.copy()
        stage2_proj[:, 1, :2, :] = stage2_proj[:, 1, :2, :] * 2
        stage3_proj = proj_matrices.copy()
        stage3_proj[:, 1, :2, :] = stage3_proj[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1":torch.from_numpy(proj_matrices).unsqueeze(0),
            "stage2":torch.from_numpy(stage2_proj).unsqueeze(0),
            "stage3":torch.from_numpy(stage3_proj).unsqueeze(0)
        }

        return {"imgs":torch.from_numpy(imgs).unsqueeze(0),
                "proj_matrices":proj_matrices_ms,
                "depth_values":torch.from_numpy(self.dvalues).unsqueeze(0)}

        # proj_matrices_ms = {
        #     "stage1":proj_matrices,
        #     "stage2":stage2_proj,
        #     "stage3":stage3_proj
        # }

        # return {"imgs":imgs,
        #         "proj_matrices":proj_matrices_ms,
        #         "depth_values":self.dvalues}

    def subCallBack(self, data):
        # add pose info
        nowStamp = data.header.stamp.to_sec()
        t = data.transform.translation
        q = data.transform.Quaternion

        tx, ty, tz = t.x, t.y, t.z
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        pose = np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float32)

        self.vTransformBuffer.append((nowStamp, pose))


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

    # local debug
    # base env params
    params.dmax = 50
    params.dmin = 0.5

    # model settings
    params.initN = 192
    params.ndepths = "128,32,8"
    params.fea_channels = 8
    params.cr_base_chs = "8,8,8"
    params.depth_inter_r = "4,2,1"
    params.grad_method = "detach"
    params.refine = False
    params.nores = False
    params.isinv = False
    params.share_cr = False

    params.modelpath = "/home/zhy/catkin_ws/src/data_publisher/scripts/checkpoint/model_000015.ckpt"

    # frame selection settings
    params.rfRange = 2
    params.nviews = 5
    params.imgpath = "/home/zhy/HDD/zhy/datasets/EuRoc/origin/index/MH01/mav0/cam0/data"
    params.depthpath = "./depths"
    params.wb = 1.0
    params.wt = 1.0
    params.conf = 0.99
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
        fea_channels=params.fea_channels,
        nores = params.nores,
        isinv = params.isinv
    )

    # load checkpoint file specified by params.loadckpt
    print("loading model {}".format(params.modelpath))
    state_dict = torch.load(params.modelpath, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    calib = np.array([[458.654, 0, 367.215],
                      [0, 457.296, 248.375],
                      [0, 0, 1]], dtype=np.float32)
    distort = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float32)

    # calib = np.array([[517.306408, 0, 318.643040],
    #                   [0, 516.469215, 255.313989],
    #                   [0, 0, 1]], dtype=np.float32)
    # distort = np.array([0, 0, 0, 0, 0], dtype=np.float32)

    tg = TransformGrabber(
        depthModel=model, calib=calib, distort=distort,
        dmin=params.dmin, dmax=params.dmax, initN=params.initN,
        imgPath=params.imgpath, 
        depthPath=params.depthpath, 
        RFRange=params.rfRange, Views=params.nviews,
        wb=params.wb, wt=params.wt, 
        relthetaMax=params.relthetaMax, conf=params.conf,
        baselinek=params.baselinek  
    )

    # test
    imgpath = "/home/zhy/HDD/zhy/datasets/EuRoc/origin/index/MH01/mav0/cam0/data"
    transform_path = "/home/zhy/HDD/zhy/datasets/EuRoc/origin/index/kf_gt_poses/MH01_test.txt"

    with open(transform_path, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        if i % 5 != 0:
            continue
        info = lines[i].strip().split(' ')
        stamp = info[0]
        # tx, ty, tz = float(info[5]), float(info[6]), float(info[7])
        # qx, qy, qz, qw = float(info[8]), float(info[9]), float(info[10]), float(info[11])

        tx, ty, tz = float(info[0]), float(info[1]), float(info[2])
        qx, qy, qz, qw = float(info[3]), float(info[4]), float(info[5]), float(info[6])
        
        Rotation_matrix = R.from_quat([qx, qy, qz, qw])
        q = Rotation_matrix.as_quat()
        qx, qy, qz, qw = q[0], q[1], q[2], q[3]
        
        pose = np.array([tx, ty, tz, qx, qy, qz, qw], dtype=np.float32)
        tg.vTransformBuffer.append((stamp, pose))

    # # sub
    # transform_stamped_sub = rospy.Subscriber(
    #     "/Mono_Inertial/Transformation", 
    #     TransformStamped, 
    #     tg.subCallBack, 
    #     queue_size=100
    # )

    # publisher
    nowIdx = tg.mRFRange

    with torch.no_grad():
        while not rospy.is_shutdown():

            # # Generate Depth and publish PointClouds
            # if len(tg.vTransformBuffer) >= nowIdx + tg.mRFRange:
            #     # select MVS Related Frames
            #     idxArray = tg.selectRF(nowIdx)

            idxArray = [nowIdx, nowIdx-2, nowIdx+2]

            # data prepare
            sample = tg.dataPrepare(idxArray)

            # depth inference
            sample_cuda = tocuda(sample)
            # outputs = tg.DepthModel(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            del sample_cuda

            nowImage = sample["imgs"][0][0].numpy()
            nowStamp = tg.vTransformBuffer[nowIdx][0]

            nowImage_vis = np.transpose(nowImage, (1, 2, 0))
            nowImage_vis = PIL.Image.fromarray((nowImage_vis * 255.0).astype(np.uint8))
            nowImage_vis.show()

            # depth selection
            depth_est = outputs["depth"][0]
            photometirc_confidence = outputs["photometric_confidence"][0]
            photo_mask = photometirc_confidence > tg.conf 

            # depth vis
            # depth_est_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_est,alpha=51), cv2.COLORMAP_AUTUMN)
            depth_est_vis = cv2.applyColorMap(cv2.convertScaleAbs(depth_est,alpha=3.5), cv2.COLORMAP_JET)
            depth_est_vis = PIL.Image.fromarray(depth_est_vis * np.expand_dims(photo_mask, axis=2))
            depth_est_vis.show()

            # # pointcloud generation
            # points = pixel_to_3DRGB(depth_est, nowImage, 1.0, tg.vcalib)
            # pointcloud_msg = xyzrgb_array_to_pointcloud2(points, nowStamp, "tmp_point")

            # # publish
            # self.PCL2Publisher.publish(pointcloud_msg)
            nowIdx += 1
    
    # rospy.spin()