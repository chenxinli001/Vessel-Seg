# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 20:29:34 2019

@author: wenaoma
"""

from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50,resnet18_2
import torch.nn.functional as F
from models.unet_parts import *
from models.layers import *

class ResUnet_illum_tran_ds3_add_conv4_sSE_cSE_up(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(8, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran7_scSE(256, 128,256)
        self.up3 = UnetBlock_tran7_scSE(256, 64,256)
        self.up4 = UnetBlock_tran7_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(8, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up2(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(8, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove() 

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up3(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(8, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(8, 8, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = F.relu(self.bn2(self.conv1x1_4(x)))
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove() 

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up4(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(8, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.conv1x1_4 = nn.Conv2d(8, 8, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn2 = nn.BatchNorm2d(8)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = F.relu(self.bn2(self.conv1x1_4(x)))
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove() 

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up2_w1(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18_2
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=False).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(8, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        #x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove() 

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_notra(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(6, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()
   
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_notra2(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(6, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()
     
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_nopre(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(5, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, pro1], dim=1)
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_nopre2(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(5, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, pro1], dim=1)
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove() 
 
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_nono(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
#        x = torch.cat([x, pro1], dim=1)
        x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_nono2(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
#        x = torch.cat([x, pro1], dim=1)
        #x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        

        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_nono3(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(3, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
#        x = torch.cat([x, pro1], dim=1)
        #x = self.conv1x1_2(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()    
     
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up_concat(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        layers = list(base_model(pretrained=True).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        #self.conv2 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3,
        #                       bias=False)

        self.conv1x1 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv1x1_2 = nn.Conv2d(6, 3, kernel_size=1, padding=0)
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(288, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(160,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(160, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)
        self.bn_trad = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
       
        pro1 = self.conv1x1_3(pro1)
        pro1 = F.relu(self.bn_trad(pro1))
        pro2 = self.conv1x1_3(pro2)
        pro2 = F.relu(self.bn_trad(pro2))
        pro3 = self.conv1x1_3(pro3)
        pro3 = F.relu(self.bn_trad(pro3))
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)    
        x = torch.cat([x, pro3], dim=1)
        x = self.up4(x, self.sfs[0].features)
        x = torch.cat([x, pro2], dim=1)

        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = torch.cat([x_out, pro1], dim=1)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2(self.sfs[2].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)

        ds2 = self.ds_deconv(self.sfs[1].features)
        ds2 = self.ds_deconv(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        ds3 = self.ds_deconv(self.sfs[0].features)
        ds3 = self.ds_conv(ds3)
        ds3 = self.bn(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()             
        
