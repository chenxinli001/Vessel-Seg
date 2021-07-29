# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 15:45:31 2019

@author: wenaoma
"""
from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50,resnet18_2,resnet18_3,resnet18_4,resnet18_5
import torch.nn.functional as F
from models.unet_parts import *
from models.layers import *

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up5(nn.Module):

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
        self.ec_block = expand_compress_block(8)
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
        x = self.ec_block(x)
       
        
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
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up6(nn.Module):

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
        self.ec_block = expand_compress_block2(8)
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
        x = self.ec_block(x)
       
        
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
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up7(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18_3
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
        self.conv1x1_3 = nn.Conv2d(8, 32, kernel_size=3, padding=1)
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
        x = F.relu(self.bn_trad(self.conv1x1_3(x)))
       
        
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
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up8(nn.Module):

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
        self.ec_block = expand_compress_block3(8)
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
        x = self.ec_block(x)
       
        
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
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up9(nn.Module):

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
        self.ec_block = expand_compress_block4(8)
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
        x = self.ec_block(x)
       
        
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
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up10(nn.Module):

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
        self.ec_block = expand_compress_block5(8)
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
        x = self.ec_block(x)
       
        
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
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up11(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18_4
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
        self.ec_block = expand_compress_block5(8)
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
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up12(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18_5
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
        self.ec_block = expand_compress_block5(8)
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
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up13(nn.Module):

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
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up14(nn.Module):

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
        self.ec_block = expand_compress_block6(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up15(nn.Module):

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
        self.ec_block = expand_compress_block7(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()  

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up16(nn.Module):

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
        self.ec_block = expand_compress_block8(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()           
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up17(nn.Module):

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
        self.ec_block = expand_compress_block4(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()            
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up18(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18_4
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
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)


        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()  

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up19(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18_5
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
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
  
        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up20(nn.Module):

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
        self.up2 = UnetBlock_tran8_scSE_bn(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE_bn(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE_bn(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up21(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        self.num_classes = num_classes
        
        layers_t1 = list(base_model(pretrained=True).children())[:cut]
        base_layers_t1 = nn.Sequential(*layers_t1)        
        self.rn_t1 = base_layers_t1
        self.sfs_t1 = [SaveFeatures(base_layers_t1[i]) for i in [2, 4]]
        self.sfs_t1.append(SaveFeatures(base_layers_t1[5][0]))

        self.up2_t1 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3_t1 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4_t1 = UnetBlock_tran8_scSE(256, 64,128)
        self.ec_block_t1 = expand_compress_block3(5)

        self.up5_1_t1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
            
        layers_t2 = list(base_model(pretrained=True).children())[:cut]
        base_layers_t2 = nn.Sequential(*layers_t2)        
        self.rn_t2 = base_layers_t2
        self.sfs_t2 = [SaveFeatures(base_layers_t2[i]) for i in [2, 4]]
        self.sfs_t2.append(SaveFeatures(base_layers_t2[5][0]))

        self.up2_t2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3_t2 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4_t2 = UnetBlock_tran8_scSE(256, 64,128)
        self.ec_block_t2 = expand_compress_block3(3)

        self.up5_1_t2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)

        self.up5_2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.up5_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.bn_1 = nn.BatchNorm2d(self.num_classes)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv2_1_t1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_deconv_1_t1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_t1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1_t2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_deconv_1_t2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_t2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_1_ = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_2_t1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_t1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_t2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_t2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2_ = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_3_t1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_3_t2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3_ = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x,y,pro1,pro2,pro3):

        x_t1 = torch.cat([x, pro1], dim=1)
        x_t1 = self.ec_block_t1(x_t1)    
        x_t1 = F.relu(self.rn_t1(x_t1))

        x_t1 = self.up2_t1(x_t1, self.sfs_t1[2].features)
        x_t1 = self.up3_t1(x_t1, self.sfs_t1[1].features)
        x_t1 = self.up4_t1(x_t1, self.sfs_t1[0].features)

        x_out_t1 = self.up5_1_t1(x_t1)
        
        x_t2 = y
        x_t2 = self.ec_block_t2(x_t2)    
        x_t2 = F.relu(self.rn_t2(x_t2))

        x_t2 = self.up2_t2(x_t2, self.sfs_t2[2].features)
        x_t2 = self.up3_t2(x_t2, self.sfs_t2[1].features)
        x_t2 = self.up4_t2(x_t2, self.sfs_t2[0].features)

        x_out_t2 = self.up5_1_t2(x_t2)
    
        x_out = torch.cat([x_out_t1, x_out_t2], dim=1)
        x_out = F.relu(self.bn1(self.up5_2(x_out)))


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1_t1 = self.ds_deconv2_1_t1(self.sfs_t1[2].features)
        ds1_t1 = self.ds_deconv_1_t1(ds1_t1)
        ds1_t1 = self.ds_deconv_1_t1_(ds1_t1)
        ds1_t2 = self.ds_deconv2_1_t2(self.sfs_t2[2].features)
        ds1_t2 = self.ds_deconv_1_t2(ds1_t2)
        ds1_t2 = self.ds_deconv_1_t2_(ds1_t2)
                
        ds1 = torch.cat([ds1_t1, ds1_t2], dim=1)
        ds1 = F.relu(self.bn2(self.ds_conv_1_(ds1)))
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2_t1 = self.ds_deconv_2_t1(self.sfs_t1[1].features)
        ds2_t1 = self.ds_deconv_2_t1_(ds2_t1)
        ds2_t2 = self.ds_deconv_2_t2(self.sfs_t2[1].features)
        ds2_t2 = self.ds_deconv_2_t2_(ds2_t2)
        
        ds2 = torch.cat([ds2_t1, ds2_t2], dim=1)
        ds2 = F.relu(self.bn3(self.ds_conv_2_(ds2)))
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3_t1 = self.ds_deconv_3_t1(self.sfs_t1[0].features)
        ds3_t2 = self.ds_deconv_3_t2(self.sfs_t2[0].features)
        
        ds3 = torch.cat([ds3_t1, ds3_t2], dim=1)
        ds3 = F.relu(self.bn4(self.ds_conv_3_(ds3)))
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs_t1: sf.remove()      
        for sf in self.sfs_t2: sf.remove()
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up22(nn.Module):

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
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up23(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut, lr_cut = [7, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34 and resnet50')

        self.num_classes = num_classes
        
        layers_t1 = list(base_model(pretrained=True).children())[:cut]
        base_layers_t1 = nn.Sequential(*layers_t1)        
        self.rn_t1 = base_layers_t1
        self.sfs_t1 = [SaveFeatures(base_layers_t1[i]) for i in [2, 4]]
        self.sfs_t1.append(SaveFeatures(base_layers_t1[5][0]))

        self.up2_t1 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3_t1 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4_t1 = UnetBlock_tran8_scSE(256, 64,128)
        self.ec_block_t1 = expand_compress_block3(5)

        self.up5_1_t1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
            
        layers_t2 = list(base_model(pretrained=True).children())[:cut]
        base_layers_t2 = nn.Sequential(*layers_t2)        
        self.rn_t2 = base_layers_t2
        self.sfs_t2 = [SaveFeatures(base_layers_t2[i]) for i in [2, 4]]
        self.sfs_t2.append(SaveFeatures(base_layers_t2[5][0]))

        self.up2_t2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3_t2 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4_t2 = UnetBlock_tran8_scSE(256, 64,128)
        self.ec_block_t2 = expand_compress_block3(3)

        self.up5_1_t2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)

        self.up5_2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.up5_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.bn_1 = nn.BatchNorm2d(self.num_classes)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv2_1_t1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_deconv_1_t1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_t1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1_t2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_deconv_1_t2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_t2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_1_ = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_2_t1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_t1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_t2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_t2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2_ = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_3_t1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_3_t2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3_ = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x,y,pro1,pro2,pro3):

        x_t1 = torch.cat([x, pro1], dim=1)
        x_t1 = self.ec_block_t1(x_t1)    
        x_t1 = F.relu(self.rn_t1(x_t1))

        x_t1 = self.up2_t1(x_t1, self.sfs_t1[2].features)
        x_t1 = self.up3_t1(x_t1, self.sfs_t1[1].features)
        x_t1 = self.up4_t1(x_t1, self.sfs_t1[0].features)

        x_out_t1 = self.up5_1_t1(x_t1)
        
        x_t2 = y
        #x_t2 = self.ec_block_t2(x_t2)    
        x_t2 = F.relu(self.rn_t2(x_t2))

        x_t2 = self.up2_t2(x_t2, self.sfs_t2[2].features)
        x_t2 = self.up3_t2(x_t2, self.sfs_t2[1].features)
        x_t2 = self.up4_t2(x_t2, self.sfs_t2[0].features)

        x_out_t2 = self.up5_1_t2(x_t2)
    
        x_out = torch.cat([x_out_t1, x_out_t2], dim=1)
        x_out = F.relu(self.bn1(self.up5_2(x_out)))


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1_t1 = self.ds_deconv2_1_t1(self.sfs_t1[2].features)
        ds1_t1 = self.ds_deconv_1_t1(ds1_t1)
        ds1_t1 = self.ds_deconv_1_t1_(ds1_t1)
        ds1_t2 = self.ds_deconv2_1_t2(self.sfs_t2[2].features)
        ds1_t2 = self.ds_deconv_1_t2(ds1_t2)
        ds1_t2 = self.ds_deconv_1_t2_(ds1_t2)
                
        ds1 = torch.cat([ds1_t1, ds1_t2], dim=1)
        ds1 = F.relu(self.bn2(self.ds_conv_1_(ds1)))
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2_t1 = self.ds_deconv_2_t1(self.sfs_t1[1].features)
        ds2_t1 = self.ds_deconv_2_t1_(ds2_t1)
        ds2_t2 = self.ds_deconv_2_t2(self.sfs_t2[1].features)
        ds2_t2 = self.ds_deconv_2_t2_(ds2_t2)
        
        ds2 = torch.cat([ds2_t1, ds2_t2], dim=1)
        ds2 = F.relu(self.bn3(self.ds_conv_2_(ds2)))
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3_t1 = self.ds_deconv_3_t1(self.sfs_t1[0].features)
        ds3_t2 = self.ds_deconv_3_t2(self.sfs_t2[0].features)
        
        ds3 = torch.cat([ds3_t1, ds3_t2], dim=1)
        ds3 = F.relu(self.bn4(self.ds_conv_3_(ds3)))
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)


        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs_t1: sf.remove()      
        for sf in self.sfs_t2: sf.remove()  
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up24(nn.Module):

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
        self.up2 = UnetBlock_tran8_scSE_bn(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE_bn(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE_bn(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up25(nn.Module):

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
        self.conv1x1_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up26(nn.Module):

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
        self.conv1x1_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.bn1 = nn.BatchNorm2d(self.num_classes)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.bn2(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv2_1(self.sfs[2].features)
        ds1 = self.ds_deconv_1(ds1)
        ds1 = self.ds_deconv_1_(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = self.ds_deconv_2(self.sfs[1].features)
        ds2 = self.ds_deconv_2_(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = self.ds_deconv_3(self.sfs[0].features)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()  

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up27(nn.Module):

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
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   
         

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up28(nn.Module):

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
        self.conv1x1_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up29(nn.Module):

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
        self.conv1x1_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.bn2(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove() 


class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up30(nn.Module):

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
        self.up2 = UnetBlock_tran8(256, 128,256)
        self.up3 = UnetBlock_tran8(256, 64,256)
        self.up4 = UnetBlock_tran8(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
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

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   
          
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up31(nn.Module):

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
        self.conv1x1_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE_noconv(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE_noconv(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE_noconv(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()                   
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up31_check(nn.Module):

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
        self.conv1x1_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE_noconv(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE_noconv(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE_noconv(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x):
        #x = self.conv1(x)
        #y = self.conv1(y)
        #x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()         


class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up34(nn.Module):

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
        self.conv1x1_5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE_1x1(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE_1x1(128, 64,256)
        self.up4 = UnetBlock_tran8_scSE_1x1(64, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()          
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up35(nn.Module):

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
        self.conv1x1_5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_1x1(256, 128,256)
        self.up3 = UnetBlock_tran8_1x1(128, 64,256)
        self.up4 = UnetBlock_tran8_1x1(64, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove() 

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up36(nn.Module):

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
        self.conv1x1_5 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran8_scSE_add(256, 128,256)
        self.up3 = UnetBlock_tran8_scSE_add(256, 64,256)
        self.up4 = UnetBlock_tran8_scSE_add(256, 64,128)
        self.ec_block = expand_compress_block3(8)
        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_1_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2_1 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv_1 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_1 = nn.BatchNorm2d(self.num_classes)

        self.ds_deconv_2 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv_2_ = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_2 = nn.BatchNorm2d(self.num_classes)
        
        self.ds_deconv_3 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv_3 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_3 = nn.BatchNorm2d(self.num_classes)

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.conv1x1_5(x_out)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv1(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv2(ds1)
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.ds_conv3(ds1)
        ds1 = self.ds_conv_1(ds1)
        ds1 = self.bn_1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv4(ds2)
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.ds_conv5(ds2)
        ds2 = self.ds_conv_2(ds2)
        ds2 = self.bn_2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.ds_conv6(ds3)
        ds3 = self.ds_conv_3(ds3)
        ds3 = self.bn_3(ds3)

        output = self.bn1(output)
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()           