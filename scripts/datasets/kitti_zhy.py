import os
import copy
import cv2
import random
import numpy as np

from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from PIL import Image

import matplotlib as mpl
import matplotlib.cm as cm
# from utils import *

class MVSDataset(Dataset):
    def __init__(self, 
                 datapath,
                 listfile,
                 mode,
                 nviews,
                 ndepths=192,
                 interval_scale=1.06):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.ext = "png"

        self.flip_array = np.array([[-1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=np.float32)

        with open(listfile, 'r') as f:
            self.files = f.readlines()
        
        assert self.mode in ["train", "val", "test"]

    def __len__(self):
        return len(self.files)

    def get_color(self, date, folder, side, vid, do_flip):
        seq_dir = os.path.join(self.datapath, date, folder)
        img_path = os.path.join(seq_dir, "image_0{}".format(side), "data", "{}.{}".format(vid, self.ext))

        img = Image.open(img_path)

        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

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

    def get_depth_ms(self, date, folder, vid, side, do_flip):
        depth_path = os.path.join(self.datapath, date, folder, "proj_depth", "groundtruth", "image_0{}".format(side), "{}.{}".format(vid, self.ext))
        
        depth = Image.open(depth_path)

        if do_flip:
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        depth = np.array(depth, dtype=np.float32) / 1000.
        depth = self.img_preprocess(depth)

        # depth_select = depth[depth > 0]
        # omax, omin = depth_select.max(), depth_select.min()

        # vmax, vmin = np.percentile(depth, 95), depth.min()
        # normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        # mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        # colormapped_im = (mapper.to_rgba(depth)[:, :, :3] * 255).astype(np.uint8)
        # im = Image.fromarray(colormapped_im)
        # im.show()

        ht, wt = depth.shape
        # ht, wt, _ = depth.shape

        depth_ms = {
            "stage1": cv2.resize(depth, (wt//4, ht//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth, (wt//2, ht//2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth
        }

        return depth_ms

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

    def __getitem__(self, index):
        # do_flip = (self.mode == "train") and random.random() > 0.5
        do_flip = False
        
        line = self.files[index].strip().split(' ')
        date, folder, side = line[0], line[1].rjust(4, '0'), line[2]
        view_ids = [line[3].rjust(10, '0'), line[4].rjust(10, '0'), line[5].rjust(10, '0')]
        folder_name = date + "_drive_" + folder + "_sync"

        side_map = {"2":"3", "3":"2"}

        imgs = []
        proj_matrices = []
        depth_values = None

        for i, vid in enumerate(view_ids):
            self.info_obtain(imgs, proj_matrices, date, folder_name, side, vid, do_flip)

            if i == 0:
                # load stereo image information
                self.info_obtain(imgs, proj_matrices, date, folder_name, side_map[side], vid, do_flip)

                # load depth range information
                depth_min, depth_max = 0, 20
                depth_interval = (depth_max - depth_min) / self.ndepths
                # depth_min, depth_interval = self.get_depth_range()
                # depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                # load depth ground truth and multi-scale information
                depth_ms = self.get_depth_ms(date, folder_name, vid, side, do_flip)
                mask = {"stage1": np.ones_like(depth_ms["stage1"]),
                        "stage2": np.ones_like(depth_ms["stage2"]),
                        "stage3": np.ones_like(depth_ms["stage3"])}

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 4
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] / 2
        stage3_pjmats = proj_matrices.copy()

        proj_matrices_ms = {
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth_values":depth_values,
                "depth":depth_ms,
                "mask": mask}


if __name__ == '__main__':
    datapath = "/home/zhy/HDD/zhy/datasets/kitti_raw"
    listfile = "./lists/kitti/train.txt"
    mode = "train"
    nviews = 3
    ndepths = 192
    interval_scale = 1.06
    dataset = MVSDataset(datapath, listfile, mode, nviews, ndepths, interval_scale)
    TrainImgLoader = DataLoader(dataset, 4, 
                                shuffle=True, num_workers=1, drop_last=True,
                                pin_memory=False)

    for batch_idx, sample in enumerate(TrainImgLoader):
        print(batch_idx)