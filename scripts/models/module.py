import torch
from torch._C import device, dtype
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import sys
sys.path.append("..")
# from utils import local_pcd

def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv2d(nn.Module):
    """Applies a 2D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.kernel_size = kernel_size
        self.stride = stride
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.stride == 2:
            h, w = list(x.size())[2:]
            y = y[:, :, :2 * h, :2 * w].contiguous()
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride, pad=1)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1, pad=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2, pad=1)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1, pad=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2, pad=1)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1, pad=1)

        self.dconv2 = nn.Sequential(
            nn.ConvTranspose3d(channels * 4, channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.Sequential(
            nn.ConvTranspose3d(channels * 2, channels, kernel_size=3, padding=1, output_padding=1, stride=2,
                               bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = F.relu(self.dconv2(conv2) + self.redir2(conv1), inplace=True)
        dconv1 = F.relu(self.dconv1(dconv2) + self.redir1(x), inplace=True)
        return dconv1


def homo_warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, Ndepth] o [B, Ndepth, H, W]
    # out: [B, C, Ndepth, H, W]
    batch, channels = src_fea.shape[0], src_fea.shape[1]
    num_depth = depth_values.shape[1]
    height, width = src_fea.shape[2], src_fea.shape[3]

    with torch.no_grad():
        # proj = (K @ T_sw) @ (K @ T_rw).inv
        proj = torch.matmul(src_proj, torch.inverse(ref_proj)) 
        rot = proj[:, :3, :3]
        trans = proj[:, :3, 3:4]

        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        
        # [ur, vr, 1]
        res = torch.stack((x, y, torch.ones_like(x)))
        res = torch.unsqueeze(res, 0).repeat(batch, 1, 1)

        # [ur, vr, 1] -> [us, vs, 1]
        res = torch.matmul(rot, res)
        res = res.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(batch, 1, num_depth, -1)
        res = res + trans.view(batch, 3, 1, 1)
        res = res[:, :2, :, :] / res[:, 2:3, :, :]

        # [us, vs, 1] -> sample grid
        res = torch.stack((res[:, 0, :, :] / ((width - 1) / 2) - 1, res[:, 1, :, :] / ((height - 1) / 2) - 1), dim=3)

    res = F.grid_sample(src_fea, res.view(batch, num_depth * height, width, 2), mode='bilinear',
                                   padding_mode='zeros', align_corners=False)
    res = res.view(batch, channels, num_depth, height, width)

    return res


class DeConv2dFuse(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, relu=True, bn=True,
                 bn_momentum=0.1):
        super(DeConv2dFuse, self).__init__()

        self.deconv = Deconv2d(in_channels, out_channels, kernel_size, stride=2, padding=1, output_padding=1,
                               bn=True, relu=relu, bn_momentum=bn_momentum)

        self.conv = Conv2d(2*out_channels, out_channels, kernel_size, stride=1, padding=1,
                           bn=bn, relu=relu, bn_momentum=bn_momentum)

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x_pre, x):
        x = self.deconv(x)
        x = torch.cat((x, x_pre), dim=1)
        x = self.conv(x)
        return x


class FeatureNet_zhy(nn.Module):
    def __init__(self, in_channels, fea_channels):
        super(FeatureNet_zhy, self).__init__()

        self.conv0 = nn.Sequential(
            Conv2d(in_channels, fea_channels, kernel_size=3, stride=1, padding=1),
            Conv2d(fea_channels, fea_channels, kernel_size=3, stride=1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(fea_channels, fea_channels*2, kernel_size=3, stride=2, padding=1),
            Conv2d(fea_channels*2, fea_channels*2, kernel_size=3, stride=1, padding=1),
            Conv2d(fea_channels*2, fea_channels*2, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(fea_channels*2, fea_channels*4, kernel_size=3, stride=2, padding=1),
            Conv2d(fea_channels*4, fea_channels*4, kernel_size=3, stride=1, padding=1),
            Conv2d(fea_channels*4, fea_channels*4, kernel_size=3, stride=1, padding=1),
        )

        self.deconv3 = DeConv2dFuse(fea_channels*4, fea_channels*2, 3)
        self.conv3 = nn.Conv2d(fea_channels*2, fea_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv4 = DeConv2dFuse(fea_channels*2, fea_channels, 3)
        self.conv4 = nn.Conv2d(fea_channels, fea_channels-1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        fea = self.conv2(conv1)
        fea = self.deconv3(conv1, fea)
        fea = self.conv3(fea)
        fea = self.deconv4(conv0, fea)
        fea = self.conv4(fea)

        gray = tensorRGB2GRAY(x)

        fea = torch.cat((gray, fea), dim=1)
        # x = torch.cat((x, conv0), dim=1)
        return fea


class FeatureNet_zhy_nores(nn.Module):
    def __init__(self, in_channels, fea_channels):
        super(FeatureNet_zhy_nores, self).__init__()

        self.conv0 = nn.Sequential(
            Conv2d(in_channels, fea_channels, kernel_size=3, stride=1, padding=1),
            Conv2d(fea_channels, fea_channels, kernel_size=3, stride=1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(fea_channels, fea_channels*2, kernel_size=3, stride=2, padding=1),
            Conv2d(fea_channels*2, fea_channels*2, kernel_size=3, stride=1, padding=1),
            Conv2d(fea_channels*2, fea_channels*2, kernel_size=3, stride=1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(fea_channels*2, fea_channels*4, kernel_size=3, stride=2, padding=1),
            Conv2d(fea_channels*4, fea_channels*4, kernel_size=3, stride=1, padding=1),
            Conv2d(fea_channels*4, fea_channels*4, kernel_size=3, stride=1, padding=1),
        )

        self.deconv3 = DeConv2dFuse(fea_channels*4, fea_channels*2, 3)
        self.conv3 = nn.Conv2d(fea_channels*2, fea_channels*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.deconv4 = DeConv2dFuse(fea_channels*2, fea_channels, 3)
        self.conv4 = nn.Conv2d(fea_channels, fea_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        fea = self.conv2(conv1)
        fea = self.deconv3(conv1, fea)
        fea = self.conv3(fea)
        fea = self.deconv4(conv0, fea)
        fea = self.conv4(fea)

        return fea


class FeatureNet(nn.Module):
    def __init__(self, base_channels, num_stage=3, stride=4, arch_mode="unet"):
        super(FeatureNet, self).__init__()
        assert arch_mode in ["unet", "fpn"], print("mode must be in 'unet' or 'fpn', but get:{}".format(arch_mode))
        print("*************feature extraction arch mode:{}****************".format(arch_mode))
        self.arch_mode = arch_mode
        self.stride = stride
        self.base_channels = base_channels
        self.num_stage = num_stage

        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.out1 = nn.Conv2d(base_channels * 4, base_channels * 4, 1, bias=False)
        self.out_channels = [4 * base_channels]

        if self.arch_mode == 'unet':
            if num_stage == 3:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)
                self.deconv2 = DeConv2dFuse(base_channels * 2, base_channels, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out3 = nn.Conv2d(base_channels, base_channels, 1, bias=False)
                self.out_channels.append(2 * base_channels)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.deconv1 = DeConv2dFuse(base_channels * 4, base_channels * 2, 3)

                self.out2 = nn.Conv2d(base_channels * 2, base_channels * 2, 1, bias=False)
                self.out_channels.append(2 * base_channels)
        elif self.arch_mode == "fpn":
            final_chs = base_channels * 4
            if num_stage == 3:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
                self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels * 2, 3, padding=1, bias=False)
                self.out3 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels * 2)
                self.out_channels.append(base_channels)

            elif num_stage == 2:
                self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)

                self.out2 = nn.Conv2d(final_chs, base_channels, 3, padding=1, bias=False)
                self.out_channels.append(base_channels)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["stage1"] = out
        if self.arch_mode == "unet":
            if self.num_stage == 3:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = self.deconv2(conv0, intra_feat)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = self.deconv1(conv1, intra_feat)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        elif self.arch_mode == "fpn":
            if self.num_stage == 3:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
                out = self.out3(intra_feat)
                outputs["stage3"] = out

            elif self.num_stage == 2:
                intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
                out = self.out2(intra_feat)
                outputs["stage2"] = out

        return outputs


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)
    
    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)

    depth = torch.sum(p * depth_values, 1)

    return depth


def f_depth_loss(inputs, depth_gt_ms, mask_ms, dlossw):
    total_loss = torch.tensor(
        0.0, 
        dtype=torch.float32, 
        device=mask_ms["stage1"].device, 
        requires_grad=False
    )

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        
        mask = mask_ms[stage_key]
        mask = (mask > 0.5) * (depth_gt > 0)

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if dlossw is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += dlossw[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss


def f_reg_loss(inputs, imgs, proj_matrices_ms, mask_ms, rgblossw):
    color_loss = torch.tensor(
        0.0, 
        dtype=torch.float32, 
        device=mask_ms["stage1"].device, 
        requires_grad=False
    )

    smooth_loss = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=mask_ms["stage1"].device,
        requires_grad=False
    )

    pos_loss = torch.tensor(
        0.0,
        dtype=torch.float32,
        device=mask_ms["stage1"].device,
        requires_grad=False
    )

    imgs = torch.unbind(imgs, 1)
    ref_img, src_imgs = imgs[0], imgs[1:]
    mask_origin = mask_ms["stage3"] > 0.5
    
    b, img_c, img_h, img_w = ref_img.shape

    ref_grad_x, ref_grad_y = compute_color_grad(ref_img)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        # 0. obtain stage depth and stage camera intrinsic K
        depth_stage = stage_inputs["depth"]
        proj_matrices_stage = proj_matrices_ms[stage_key]

        # 1. unbind and data pre process
        proj_matrices_stage = torch.unbind(proj_matrices_stage, 1)
        ref_proj, src_projs = proj_matrices_stage[0], proj_matrices_stage[1:]

        T_rw = ref_proj[:, 0, :4, :4]
        K_r = ref_proj[:, 1, :3, :3]

        b, depth_h, depth_w = depth_stage.shape
        depth_stage = depth_stage.unsqueeze(1)

        if depth_h != img_h or depth_w != img_w:
            depth_stage = F.interpolate(
                depth_stage, 
                [img_h, img_w], 
                mode="bilinear", 
                align_corners=False
            )

        stage_scale = img_w / depth_w

        K_r[:, 0, :] = K_r[:, 0, :] * stage_scale
        K_r[:, 1, :] = K_r[:, 1, :] * stage_scale

        # 2. img pixel to ref coordinate
        xyz_r = depth_stage.view(b, 1, -1) * pixel2one(b, img_h, img_w, K_r)

        # 3. compute ssim and color loss
        color_loss_tmp = compute_color_loss(
            xyz_r, ref_img, src_imgs, mask_origin.unsqueeze(1),
            img_h, img_w, stage_scale,
            src_projs, T_rw
        )

        # 4. compute smooth loss
        smooth_loss_tmp = compute_smooth_loss(
            depth_stage, ref_grad_x, ref_grad_y, mask_origin
        )

        # 5. compute pos loss
        pos_loss_tmp = compute_pos_loss(
            xyz_r.view(b, 3, img_h, img_w), ref_grad_x, ref_grad_y, mask_origin
        )

        # 6. weight for stage
        if rgblossw is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            color_loss += rgblossw[stage_idx] * color_loss_tmp
            smooth_loss += rgblossw[stage_idx] * smooth_loss_tmp
            pos_loss += rgblossw[stage_idx] * pos_loss_tmp
        else:
            color_loss += color_loss_tmp
            smooth_loss += smooth_loss_tmp
            pos_loss += pos_loss_tmp

    return color_loss, smooth_loss, pos_loss


ssim = SSIM()


def tensorRGB2GRAY(img):
    return (img[:, 0] * 0.299 + img[:, 1] * 0.587 + img[:, 2] * 0.114).unsqueeze(1)


def compute_color_grad(img):
    b, _, h, w = img.shape

    grad_img_x = torch.mean(torch.abs(img[:, :, 1:h-1, 0:w-2] - img[:, :, 1:h-1, 2:w]) * 0.5, 1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, 0:h-2, 1:w-1] - img[:, :, 2:h, 1:w-1]) * 0.5, 1, keepdim=True)

    return grad_img_x, grad_img_y


def compute_reprojection_loss(pred, target):
    # return 
    abs_diff = torch.abs(pred - target)
    color_loss = abs_diff.mean(1, True)
    # return color_loss
    ssim_loss = ssim(pred, target).mean(1, True) 
    return 0.85 * ssim_loss + 0.15 * color_loss


def compute_color_loss(
    xyz_r, ref_img, src_imgs, ref_mask, 
    img_h, img_w, 
    stage_scale, src_projs, T_rw
):
    warp_loss = None
    direct_loss = None

    for (src_proj, src_img) in zip(src_projs, src_imgs):
        T_sw = src_proj[:, 0, :4, :4]
        K_s = src_proj[:, 1, :3, :3]

        K_s[:, 0, :] = K_s[:, 0, :] * stage_scale
        K_s[:, 1, :] = K_s[:, 1, :] * stage_scale

        T_sr = torch.matmul(T_sw, torch.inverse(T_rw))
        xyz_s = torch.matmul(T_sr[:, :3, :3], xyz_r) + T_sr[:, :3, 3:]
        sample_rins = cam2pixel(xyz_s, K_s, img_h, img_w)

        warp_ref_img = F.grid_sample(
            src_img,
            sample_rins,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False
        )

        if warp_loss is None:
            warp_loss = compute_reprojection_loss(warp_ref_img, ref_img)
        else:
            warp_loss = torch.cat((warp_loss, compute_reprojection_loss(warp_ref_img, ref_img)), dim=1)

        if direct_loss is None:
            direct_loss = compute_reprojection_loss(src_img, ref_img)
        else:
            direct_loss = torch.cat((direct_loss,compute_reprojection_loss(src_img, ref_img)),dim=1)

    to_optimise, _ = torch.min(torch.cat((warp_loss, direct_loss), dim=1), dim=1, keepdim=True)
    color_loss = torch.sum(to_optimise * ref_mask) / torch.sum(ref_mask)

    return color_loss


def compute_smooth_loss(depth, grad_img_x, grad_img_y, mask=None):
    b, _, h, w = depth.shape
    invdepth = 1.0 / depth

    if mask is not None:
        mask_now = mask[:, 1:h-1, 1:w-1].unsqueeze(1)

    grad_depth_x = torch.abs(invdepth[:, :, 1:h-1, 0:w-2] - invdepth[:, :, 1:h-1, 2:w]) * 0.5
    grad_depth_y = torch.abs(invdepth[:, :, 0:h-2, 1:w-1] - invdepth[:, :, 2:h, 1:w-1]) * 0.5

    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)

    if mask is not None:
        loss = torch.sum((grad_depth_x + grad_depth_y) * mask_now) / torch.sum(mask_now)
    else:
        loss = torch.mean(grad_depth_x + grad_depth_y)
    
    return loss


def compute_pos_loss(xyz_r, grad_img_x, grad_img_y, mask=None):
    b, _, h, w = xyz_r.shape

    if mask is not None:
        mask_now = mask[:, 1:h-1, 1:w-1].unsqueeze(1)
    
    #[b, 3, h, w] -> [b, h-2, w-2]
    loss_l = (xyz_r[:, 0, 1:h-1, 0:w-2] - xyz_r[:, 0, 1:h-1, 1:w-1]) / xyz_r[:, 2, 1:h-1, 1:w-1]
    loss_r = (xyz_r[:, 0, 1:h-1, 1:w-1] - xyz_r[:, 0, 1:h-1, 2:w]) / xyz_r[:, 2, 1:h-1, 1:w-1]
    loss_lr = loss_l + loss_r

    loss_l = torch.clamp(loss_l, min=0.0)
    loss_r = torch.clamp(loss_r, min=0.0)
    loss_lr = torch.clamp(loss_lr, min=0.0)

    loss_t = (xyz_r[:, 1, 0:h-2, 1:w-1] - xyz_r[:, 1, 1:h-1, 1:w-1]) / xyz_r[:, 2, 1:h-1, 1:w-1]
    loss_b = (xyz_r[:, 1, 1:h-1, 1:w-1] - xyz_r[:, 1, 2:h, 1:w-1]) / xyz_r[:, 2, 1:h-1, 1:w-1]
    loss_tb = loss_t + loss_b

    loss_t = torch.clamp(loss_t, min=0.0)
    loss_b = torch.clamp(loss_b, min=0.0)
    loss_tb = torch.clamp(loss_tb, min=0.0)

    x_only_loss = loss_l + loss_r + 2.0 * loss_lr
    y_only_loss = loss_t + loss_b + 2.0 * loss_tb

    loss = x_only_loss * torch.exp(-grad_img_x) + y_only_loss * torch.exp(-grad_img_y)

    if mask is not None:
        loss = torch.sum(loss * mask_now) / torch.sum(mask_now)
    else:
        loss = torch.mean(loss)

    return loss


def get_invdepth_range_samples(
    cur_depth, ndepth, invdepth_interval, 
    device, dtype, shape, eps=1e-12
):
    #shape: (B, H, W)
    #cur_depth: (B, H, W) or None
    #return depth_range_samples: (B, D, H, W)
    if cur_depth.dim() == 2:
        invdepth_min = 1.0 / (cur_depth[:, -1] + eps)
        invdepth_max = 1.0 / (cur_depth[:, 0] + eps)

        new_interval = (invdepth_max - invdepth_min) / (ndepth - 1)
        
        depth_range_samples = invdepth_min.unsqueeze(1)
        depth_range_samples = depth_range_samples +\
             torch.arange(0, ndepth, device=device, dtype=dtype, requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)
        depth_range_samples = 1.0 / depth_range_samples    

        # [B, D] -> [B, D, H, W]
        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) 
    else:
        cur_invdepth = 1.0 / (cur_depth + eps)

        cur_invdepth_min = (cur_invdepth - ndepth * invdepth_interval / 2.0).clamp(min=eps)
        cur_invdepth_max = (cur_invdepth + ndepth * invdepth_interval / 2.0)

        assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
        new_interval = (cur_invdepth_max - cur_invdepth_min) / (ndepth - 1)  # (B, H, W)

        depth_range_samples = cur_invdepth_min.unsqueeze(1) +\
            torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype, requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1)

        depth_range_samples = 1.0 / depth_range_samples

    return depth_range_samples


def get_depth_range_samples(
    cur_depth, ndepth, depth_interval, 
    device, dtype, shape
):
    if cur_depth.dim() == 2:
        cur_depth_min = cur_depth[:, 0]  # (B,)
        cur_depth_max = cur_depth[:, -1]
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)  # (B, )

        depth_range_samples = cur_depth_min.unsqueeze(1) +\
             (torch.arange(0, ndepth, device=device, dtype=dtype, requires_grad=False).reshape(1, -1) * new_interval.unsqueeze(1)) #(B, D)

        depth_range_samples = depth_range_samples.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, shape[1], shape[2]) #(B, D, H, W)
    else:
        cur_depth_min = (cur_depth - ndepth / 2 * depth_interval).clamp(min=0.0)
        cur_depth_max = (cur_depth + ndepth / 2 * depth_interval)

        assert cur_depth.shape == torch.Size(shape), "cur_depth:{}, input shape:{}".format(cur_depth.shape, shape)
        new_interval = (cur_depth_max - cur_depth_min) / (ndepth - 1)

        depth_range_samples = cur_depth_min.unsqueeze(1) +\
            torch.arange(0, ndepth, device=cur_depth.device, dtype=cur_depth.dtype, requires_grad=False).reshape(1, -1, 1, 1) * new_interval.unsqueeze(1)

    return depth_range_samples


def pixel2one(batch, height, width, K):
    with torch.no_grad():
        K_inv = torch.inverse(K)
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=K.device),
                               torch.arange(0, width, dtype=torch.float32, device=K.device)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height*width), x.view(height*width)

        xy1 = torch.stack((x, y, torch.ones_like(x)))
        xy1 = torch.unsqueeze(xy1, 0).repeat(batch, 1, 1)
        xy1 = torch.matmul(K_inv, xy1)

    return xy1


def cam2pixel(cam, K, height, width):
    batch = cam.shape[0]
    #print(cam)
    cam_points = torch.matmul(K[:, :3, :3], cam)  # [B, 3, H*W]
    #print(cam_points)
    pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + 1e-7)
    pix_coords = pix_coords.view(batch, 2, height, width)

    sample_pix = pix_coords.permute(0, 2, 3, 1)
    sample_pix[..., 0] /= width - 1
    sample_pix[..., 1] /= height - 1
    sample_pix = 2 * sample_pix - 1

    return sample_pix