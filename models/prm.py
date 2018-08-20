# -*- coding:utf-8 -*-
# author:Jackiechen
# 2018/8/19 16:59
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class DownSample(nn.Module):
    def __init__(self,scale):
        super(DownSample, self).__init__()
        self.scale = scale

    def forward(self, x):
        sample = F.interpolate(x,scale_factor=self.scale)
        return sample

class BnResidualConv1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BnResidualConv1,self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(self.in_channels,self.out_channels,kernel_size=1,padding=0)

    def forward(self, x):
        x = self.bn(x)
        return self.conv(F.relu(x))

class BnResidualConv3(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(BnResidualConv3,self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)

        self.bn = nn.BatchNorm2d(self.in_channels)
        self.conv = nn.Conv2d(self.in_channels,self.out_channels,kernel_size=3,padding=1)

    def forward(self, x):
        x = self.bn(x)
        return self.conv(F.relu(x))

class PRM(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(PRM,self).__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.reo1 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo2 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo3 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo4 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo5 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels))

        # maxpool最大下采样到一半的size,选三个size做残差连接
        self.down1 = DownSample(scale=pow(2,-1))       # 二倍下采样 32
        self.down2 = DownSample(scale=pow(2,-0.75))
        self.down3 = DownSample(scale=pow(2,-0.5))
        self.down4 = DownSample(scale=pow(2,-0.25))

        self.ret1 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        self.ret2 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        self.ret3 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        self.ret4 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)

        # 对resize的部分重新上采样，之后要拼接
        # self.up1 = nn.Upsample(scale_factor=2)
        # self.up2 = nn.Sequential(nn.Upsample(scale_factor=2),
        #                          nn.Transpose2d(int(self.out_channels/2),int(self.out_channels/2),kernel_size=9))
        # self.up3 = nn.Upsample(size=(64,64),mode='bilinear',align_corners=True)
        # self.up4 = nn.Upsample(size=(64,64),mode='bilinear',align_corners=True)

        self.concat = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1)
        # self.conv = nn.Conv2d(self.out_channels,self.in_channels,kernel_size=1)
        # self.seq1 = nn.Sequential(self.reo1,self.down1,self.ret1,self.up1)
        # self.concat = torch.cat()

    def forward(self, x):
        identity = self.reo5(x)
        size = identity.size()[-2],identity.size()[-1]
        # 各支路单元信息
        # 先做BN + relu + 1x1卷积
        scale1 = self.reo1(x)
        # scale2 = self.reo2(x)
        # scale3 = self.reo3(x)
        scale2 = self.reo1(x)
        scale3 = self.reo1(x)
        scale4 = self.reo1(x)

        ratio1 = F.interpolate(self.ret1(self.down1(scale1)),size=size)
        ratio2 = F.interpolate(self.ret2(self.down2(scale2)),size=size)
        ratio3 = F.interpolate(self.ret3(self.down3(scale3)),size=size)
        ratio4 = F.interpolate(self.ret4(self.down4(scale4)),size=size)

        tmp_ret = ratio1+ratio2+ratio3+ratio4
        add = self.concat(tmp_ret)
        ret = identity + add
        # ret相比原输入，size不变
        return ret
        #return identity+ratio1+ratio2+ratio3
