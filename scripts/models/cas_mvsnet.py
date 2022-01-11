import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *

Align_Corners_Range = False

class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.grads = []

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None):
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)
        num_views = len(features)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume 
        volume_sq_sum = ref_volume ** 2
        
        del ref_volume

        for src_fea, src_proj in zip(src_features, src_projs):
            #warpped features
            src_proj_new = src_proj[:, 0].clone()
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone()
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
        
            warped_volume = homo_warping(src_fea, src_proj_new, ref_proj_new, depth_values)

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume

        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = cost_regularization(volume_variance)
        
        prob_volume_pre = cost_reg.squeeze(1)

        if prob_volume_init is not None:
            prob_volume_pre += prob_volume_init
        
        prob_volume = F.softmax(prob_volume_pre, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            depth_index = depth_index.clamp(min=0, max=num_depth-1)
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)
        
        return {"depth": depth,  "photometric_confidence": photometric_confidence}


class CascadeMVSNet(nn.Module):
    def __init__(
        self, 
        refine = False, 
        ndepths = [48, 32, 8], 
        depth_interals_ratio = [4, 2, 1], 
        share_cr = False,
        grad_method = "detach", 
        arch_mode = "fpn", 
        cr_base_chs = [8, 8, 8],
        fea_channels = 3,
        nores = True,
        isinv = False
    ):
        super(CascadeMVSNet, self).__init__()

        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        self.isinv = isinv

        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
              depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
            "stage1":{
                "scale": 4.0,
            },
            "stage2": {
                "scale": 2.0,
            },
            "stage3": {
                "scale": 1.0,
            }
        }

        if nores:
            self.feature = FeatureNet_zhy_nores(in_channels=3, fea_channels=fea_channels)
        else:
            self.feature = FeatureNet_zhy(in_channels=3, fea_channels=fea_channels)

        # self.feature = FeatureNet(base_channels=8, stride=4, num_stage=self.num_stage, arch_mode=self.arch_mode)
        self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=fea_channels, base_channels=8) for _ in range(self.num_stage)])

        if self.refine:
            self.refine_network = RefineNet()
            
        self.DepthNet = DepthNet()

    def forward(self, imgs, proj_matrices, depth_values):
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())

        # need depth_values.size(1) > 1
        if self.isinv:
            inv_depth_interval = (1.0/depth_min - 1.0/depth_max) / (depth_values.size(1) - 1)
        else:
            depth_interval = (depth_max - depth_min) / (depth_values.size(1) - 1)

        batch_size, nviews, channel_size, h, w = imgs.shape
        data_type, data_device = imgs[0, 0].dtype, imgs[0, 0].device

        # 0. feature extraction
        features = []
        for i in range(imgs.size(1)):
            img = imgs[:, i]
            feat = {}

            for stage_idx in range(self.num_stage):
                scale_stage = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
                img_stage = F.interpolate(
                    img, 
                    size = (h//int(scale_stage), w//int(scale_stage)), 
                    mode = 'bilinear',
                    align_corners = False
                )
                feat["stage{}".format(stage_idx+1)] = self.feature(img_stage)
            
            features.append(feat)

        # imgs = torch.unbind(imgs, dim=1)

        outputs = {}
        depth, cur_depth = None, None

        # 1. casecade prediction
        for stage_idx in range(self.num_stage):

            # 1-0. load stage info
            scale_stage = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            fea_stage = [feat["stage{}".format(stage_idx+1)] for feat in features]
            # proj_stage = proj_matrices
            proj_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            ndepth_stage = self.ndepths[stage_idx] # [48, 32, 8]
            dratio_stage = self.depth_interals_ratio[stage_idx] # [4, 2, 1]
            h_stage = h // int(scale_stage)
            w_stage = w // int(scale_stage)
            cost_reg_stage = self.cost_regularization[stage_idx]

            # 1-1. initiate depth
            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth

                cur_depth = F.interpolate(
                    cur_depth.unsqueeze(1),
                    [h_stage, w_stage],
                    mode = 'bilinear',
                    align_corners = False).squeeze(1)
            else:
                cur_depth = depth_values

            # 1-2. reset depth range
            if self.isinv:
                depth_range_samples = get_invdepth_range_samples(
                    cur_depth = cur_depth,
                    ndepth = ndepth_stage,
                    invdepth_interval = dratio_stage * inv_depth_interval,
                    device = data_device,
                    dtype = data_type,
                    shape = [batch_size, h_stage, w_stage],
                )
            else:
                depth_range_samples = get_depth_range_samples(
                    cur_depth = cur_depth,
                    ndepth = ndepth_stage,
                    depth_interval = dratio_stage * depth_interval,
                    device = data_device,
                    dtype = data_type,
                    shape = [batch_size, h_stage, w_stage]
                )
            
            # 1-3. stage depth prediction
            outputs_stage = self.DepthNet(
                features = fea_stage,
                proj_matrices = proj_stage,
                depth_values = depth_range_samples,
                num_depth = ndepth_stage,
                cost_regularization = cost_reg_stage
            )

            depth = outputs_stage['depth']

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        # 2. depth map refinement
        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return outputs