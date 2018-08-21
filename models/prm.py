# -*- coding:utf-8 -*-
# author:Jackiechen
# 2018/8/19 16:59
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Concat(nn.Module):
    # for PRM-C
    def __init__(self):
        super(Concat,self).__init__()
    
    def forward(self, input):
        return torch.cat(input,1)

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
        # When choose PRM-A，uncomment reo2-reo4
        self.reo2 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo3 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo4 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels/2))
        self.reo5 = BnResidualConv1(in_channels=self.in_channels,out_channels=int(self.out_channels))

        # downsample to multi-scale
        self.down1 = DownSample(scale=pow(2,-1))      
        self.down2 = DownSample(scale=pow(2,-0.75))
        self.down3 = DownSample(scale=pow(2,-0.5))
        self.down4 = DownSample(scale=pow(2,-0.25))

        self.ret1 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        self.ret2 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        self.ret3 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        self.ret4 = BnResidualConv3(in_channels=int(self.out_channels/2),out_channels=self.out_channels)
        # PRM-B
        self.smooth = BnResidualConv1(in_channels=self.out_channels,out_channels=self.out_channels)
        # PRM-C
        # self.smooth = BnResidualConv1(in_channels=self.out_channels*4,out_channels=self.out_channels)

    def forward(self, x):
        identity = self.reo5(x)
        size = identity.size()[-2],identity.size()[-1]
        # multi-scale information
        # BN + relu + 1x1 conv
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
        # PRM-B
        tmp_ret = ratio1+ratio2+ratio3+ratio4
        # PRM-C,replace smooth with PRM-C's smooth layer
        # tmp_ret = torch.cat((ratio1,ratio2,ratio3,ratio4),1)
        smooth = self.smooth(tmp_ret)
        ret = identity + smooth
        # size equal
        return ret
        #return identity+ratio1+ratio2+ratio3
