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


class KITTIraw_Test():
    def __init__(self, datapath):
        self.datapath = datapath
        self.ext = "png"

    def get_color(self, date, folder, side, vid, do_flip):
        seq_dir = os.path.join(self.datapath, date, folder)
        img_path = os.path.join(seq_dir, "image_0{}".format(side), "data", "{}.{}".format(vid, self.ext))

        img = PIL.Image.open(img_path)

        if do_flip:
            img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

        img = img.convert('RGB')

        return np.array(img, dtype=np.float32) / 255.

    def get_pose(self, date, folder, side, vid):
        gt_pose_path = os.path.join(self.datapath, date, folder, "poses", "image_0{}".format(side), "{}.txt".format(vid))

        T_wc = np.loadtxt(gt_pose_path, dtype=np.float32)

        return T_wc

    def get_intrinsics(self, date, side):
        calib_path = os.path.join(self.datapath, date, "calib_cam_to_cam.txt")

        with open(calib_path, 'r') as f:
            lines = f.readlines()
            dict_name = "P_rect_0{}".format(side)

            K = np.zeros((3, 3), dtype=np.float32)
            K[2, 2] = 1
            t = np.zeros((3, 1), dtype=np.float32)

            for line in lines:
                name = line.strip().split(':')[0]
                if name == dict_name:
                    data = line.strip().split(':')[1]
                    data_list = data.strip().split(' ')
                    K[0, 0], K[0, 2] = float(data_list[0]), float(data_list[2])
                    K[1, 1], K[1, 2] = float(data_list[5]), float(data_list[6])
                    t[0], t[1], t[2] = float(data_list[3]), float(data_list[7]), float(data_list[11])
        
        return K, t

    def img_preprocess(self, img):
        h, w = img.shape[0], img.shape[1]
        wt, ht = (w // 32) * 32, (h // 32) * 32
        
        ws, hs = (w-wt) // 2, (h-ht) // 2
        img_crop = img[hs: hs+ht, ws:ws+wt]

        return img_crop

    def info_obtain(self, imgs, projs, date, folder, side, vid, do_flip):
        # This place should be changed for T_cw be used in the train.py
        img = self.get_color(date, folder, side, vid, do_flip)
        T_wc = self.get_pose(date, folder, side, vid)
        K, t = self.get_intrinsics(date, side)

        h, w, _ = img.shape
        img = self.img_preprocess(img)
        ht, wt, _ = img.shape
        
        K[0, 2] = K[0, 2] - (w - wt) // 2
        K[1, 2] = K[1, 2] - (h - ht) // 2

        if do_flip:
            T_wc = self.flip_array @ T_wc @ self.flip_array

        proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
        proj_mat[0, :4, :4] = np.linalg.inv(T_wc)
        proj_mat[1, :3, :3] = K

        projs.append(proj_mat)
        imgs.append(img)


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
        T_nw = nPose
        T_rfw = rfPose

        T_rel = T_rfw @ np.linalg.inv(T_nw)

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
        # retIdxArray = [nIdx]
        # cmpArray = []

        # for i in range(-self.mRFRange, self.mRFRange+1):
        #     if i == 0:
        #         continue
        #     tmpPose = self.vTransformBuffer[nIdx+i][1]
        #     tmpScore = self.calRFScore(nPose, tmpPose)

        #     cmpArray.append((tmpScore, nIdx+i))

        # cmpArray.sort(reverse=True)

        # for i in range (self.mViews-1):
        #     retIdxArray.append(cmpArray[i][1])

        retIdxArray = [nIdx, nIdx-2, nIdx+2]

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

            img = PIL.Image.open(os.path.join(self.mImgPath, "image_2", stamp+".png"))
            img = img.convert('RGB')
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            img = img.astype(np.float32) / 255.

            # img = np.array(img, dtype=np.float32) / 255.

            # img = cv2.undistort(img, self.vcalib, self.vdistort, None)
            # img_undist_vis = PIL.Image.fromarray((img * 255).astype(np.uint8))
            # img_undist_vis.show()

            if len(img.shape) == 2:
                img = np.stack((img,)*3, axis=-1)

            # img crop form topleft, do not need to change intrinsic
            h, w, _ = img.shape

            wresize = w // 32 * 32
            hresize = h // 32 * 32
            img = img[:hresize, :wresize]

            Tcw = np.linalg.inv(pose)

            intrinsic = self.vcalib.copy()
            intrinsic[:2, :] /= 4.0

            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = Tcw
            proj_mat[1, :3, :3] = intrinsic
            proj_matrices.append(proj_mat)

            imgs.append(img)

            if i == 0:
                stamp = str((self.vTransformBuffer[idx][0]))
                pose = self.vTransformBuffer[idx][1]

                img = PIL.Image.open(os.path.join(self.mImgPath, "image_3", stamp+".png"))
                img = img.convert('RGB')
                img = np.array(img, dtype=np.float32) / 255.

                # img = cv2.undistort(img, self.vcalib, self.vdistort, None)
                # img_undist_vis = PIL.Image.fromarray((img * 255).astype(np.uint8))
                # img_undist_vis.show()

                if len(img.shape) == 2:
                    img = np.stack((img,)*3, axis=-1)

                # img crop form topleft, do not need to change intrinsic
                h, w, _ = img.shape

                wresize = w // 32 * 32
                hresize = h // 32 * 32
                img = img[:hresize, :wresize]

                T_32 = np.array([[1.0, 0.0, 0.0, -0.5323318],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

                Tcw = np.linalg.inv(pose)
                Tcw = T_32 @ Tcw

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
    params.dmax = 80.0
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
    params.imgpath = "/home/zhy/HDD/zhy/datasets/kitti_odo/dataset/sequences/10"
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

    # load checkpoint file specified by params.loadckpt
    print("loading model {}".format(params.modelpath))
    state_dict = torch.load(params.modelpath, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    calib = np.array([[718.856, 0, 607.1928],
                          [0, 718.856, 185.2157],
                          [0, 0, 1]], dtype=np.float32)
    distort = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05], dtype=np.float32)

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

    # kitti_odo
    outputpath = "/home/zhy/HDD/zhy/tmp"
    transform_path = "/home/zhy/HDD/zhy/datasets/kitti_odo/dataset/poses/10.txt"

    with open(transform_path, 'r') as f:
        lines = f.readlines()

    for i in range(len(lines)):
        name = str(i).rjust(6, '0')
        info = lines[i].strip().split(' ')

        tmp_pose = np.eye(4, dtype=np.float32)
        tmp_pose[0,0], tmp_pose[0,1], tmp_pose[0,2], tmp_pose[0,3] = \
            info[0], info[1], info[2], info[3]
        tmp_pose[1,0], tmp_pose[1,1], tmp_pose[1,2], tmp_pose[1,3] = \
            info[4], info[5], info[6], info[7]
        tmp_pose[2,0], tmp_pose[2,1], tmp_pose[2,2], tmp_pose[2,3] = \
            info[8], info[9], info[10], info[11]

        tg.vTransformBuffer.append((name, tmp_pose))

    nowIdx = 2
    frame_id = "vicon"
    rate = rospy.Rate(5)

    with torch.no_grad():
        while not rospy.is_shutdown() and nowIdx < len(tg.vTransformBuffer)-2:
            # select MVS Related Frames
            idxArray = tg.selectRF(nowIdx)

            # data prepare
            sample = tg.dataPrepare(idxArray)

            # depth inference
            sample_cuda = tocuda(sample)
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
            outputs = tensor2numpy(outputs)
            del sample_cuda

            nowImage = np.transpose(sample["imgs"][0][0].numpy(), (1,2,0))
            nowImage = (nowImage * 255.0).astype(np.uint8)
            nowImageIdx = tg.vTransformBuffer[nowIdx][0]
            timeStamp = rospy.Time.now()

            # nowImage_vis = np.transpose(nowImage, (1, 2, 0))
            # nowImage_vis = PIL.Image.fromarray((nowImage_vis * 255.0).astype(np.uint8))
            # nowImage_vis.show()

            # depth selection
            depth_est = outputs["depth"][0]
            photometirc_confidence = outputs["photometric_confidence"][0]
            h, w = photometirc_confidence.shape
            photo_mask = photometirc_confidence > tg.conf
            photo_mask[h//2:, :] = False

            # # depth vis
            # vmin, vmax = depth_est.min(), np.percentile(depth_est, 95)
            # normalizer = mpl.colors.Normalize(vmin=-vmax, vmax=-vmin)
            # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            # depth_est_vis = (mapper.to_rgba(-depth_est.squeeze())[:, :, :3] * 255).astype(np.uint8)
            # depth_est_vis = PIL.Image.fromarray(depth_est_vis)
            # depth_est_vis.show()
            # depth_est_vis.save(os.path.join(outputpath, nowImageIdx+".png"))

            depth_est = depth_est * photo_mask

            # pointcloud generation
            points = pixel_to_3DRGB(depth_est, nowImage, 1.0, tg.vcalib)
            pointcloud_msg = xyzrgb_array_to_pointcloud2(points, timeStamp, "tmp_point")

            Tcw = sample["proj_matrices"]["stage3"][0,0,0] # Tcw
            pose = np.linalg.inv(Tcw)
            rotation = R.from_matrix(pose[:3, :3])
            q = rotation.as_quat()
            tx, ty, tz = pose[0, 3], pose[1, 3], pose[2, 3]
            qx, qy, qz, qw = q[0], q[1], q[2], q[3]

            transform_stamped_msg = tq_to_TransformStamped(
                Vector3(x=tx, y=ty, z=tz),
                Quaternion(x=qx, y=qy, z=qz, w=qw),
                child_frame_id="kinect", 
                stamp=timeStamp, 
                frame_id=frame_id
            )

            # publish
            transform_stamped_pub.publish(transform_stamped_msg)
            pointcloud2_pub.publish(pointcloud_msg)

            rate.sleep()

            nowIdx += 1

    rospy.spin()

    # # kitti raw

    # kitti_raw_test = KITTIraw_Test(params.imgpath)
    # raw_date = "2011_09_26"
    # raw_folder = "0029"
    # folder_name = raw_date + "_drive_" + raw_folder + "_sync"
    # side = "2"
    # side_map = {"2":"3", "3":"2"}

    # seq_dir = os.path.join(params.imgpath, raw_date, folder_name, "image_0{}".format(side), "data")
    # imglist = os.listdir(seq_dir)
    # imglist.sort()
    # outdir = "/home/zhy/HDD/zhy/tmp"

    # with torch.no_grad():
    #     for i in (2, len(imglist)-2):
    #         imgs = []
    #         projs = []
    #         vids = [i, i-2, i+2]
    #         depth_values = None

    #         ref_name = imglist[i].split('.')[0]

    #         for vid in vids:
    #             imgfile = imglist[i]
    #             img_name, img_ext = imgfile.split('.')
    #             kitti_raw_test.info_obtain(imgs, projs, raw_date, folder_name, side, img_name, False)

    #             if vid == vids[0]:
    #                 # load stereo image information
    #                 kitti_raw_test.info_obtain(imgs, projs, raw_date, folder_name, side_map[side], img_name, False)

    #                 # load depth range information
    #                 depth_min, depth_max = params.dmin, params.dmax
    #                 depth_interval = (depth_max - depth_min) / params.initN
    #                 depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

    #         imgs = np.stack(imgs).transpose([0, 3, 1, 2])
    #         proj_matrices = np.stack(projs)

    #         stage1_pjmats = proj_matrices.copy()
    #         stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4
    #         stage2_pjmats = proj_matrices.copy()
    #         stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2
    #         stage3_pjmats = proj_matrices.copy()

    #         proj_matrices_ms = {
    #             "stage1": torch.from_numpy(stage1_pjmats).unsqueeze(0),
    #             "stage2": torch.from_numpy(stage2_pjmats).unsqueeze(0),
    #             "stage3": torch.from_numpy(stage3_pjmats).unsqueeze(0)
    #         }

    #         sample = {
    #             "imgs": torch.from_numpy(imgs).unsqueeze(0),
    #             "proj_matrices": proj_matrices_ms,
    #             "depth_values":torch.from_numpy(depth_values).unsqueeze(0)
    #         }

    #         sample_cuda = tocuda(sample)
    #         outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    #         outputs = tensor2numpy(outputs)
    #         del sample_cuda

    #         nowImage = sample["imgs"][0][0].numpy()
    #         nowImage_vis = np.transpose(nowImage, (1, 2, 0))
    #         nowImage_vis = PIL.Image.fromarray((nowImage_vis * 255.0).astype(np.uint8))
    #         nowImage_vis.show()

    #         depth_est = outputs["depth"][0]
    #         vmin, vmax = depth_est.min(), np.percentile(depth_est, 95)
    #         normalizer = mpl.colors.Normalize(vmin=-vmax, vmax=-vmin)
    #         mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    #         depth_est_vis = (mapper.to_rgba(-depth_est.squeeze())[:, :, :3] * 255).astype(np.uint8)
    #         depth_est_vis = PIL.Image.fromarray(depth_est_vis)
    #         depth_est_vis.show()
    #         depth_est_vis.save(os.path.join(outdir, ref_name+".png"))




# # # # sub
# # # transform_stamped_sub = rospy.Subscriber(
# # #     "/Mono_Inertial/Transformation", 
# # #     TransformStamped, 
# # #     tg.subCallBack, 
# # #     queue_size=100
# # # )

# # # publisher
# # nowIdx = 2