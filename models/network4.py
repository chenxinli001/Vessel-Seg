# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 11:44:59 2019

@author: wenaoma
"""

from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50,resnet18_2,resnet18_3,resnet18_4,resnet18_5
import torch.nn.functional as F
from models.unet_parts import *
from models.layers import *

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up48(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block1(64)
        self.out2 = output_block1(64)
        self.out3 = output_block1(64)
        self.out4 = output_block1(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove() 
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up49(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block2(64)
        self.out2 = output_block2(64)
        self.out3 = output_block2(64)
        self.out4 = output_block2(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()  


class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up50(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block3(64)
        self.out2 = output_block3(64)
        self.out3 = output_block3(64)
        self.out4 = output_block3(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up51(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block4(64)
        self.out2 = output_block4(64)
        self.out3 = output_block4(64)
        self.out4 = output_block4(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()  


class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up52(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block4(64)
        self.out2 = output_block4(64)
        self.out3 = output_block4(64)
        self.out4 = output_block2(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()   
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up53(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block3(64)
        self.out2 = output_block3(64)
        self.out3 = output_block3(64)
        self.out4 = output_block1(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()          

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up49_wopro(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(6)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block2(64)
        self.out2 = output_block2(64)
        self.out3 = output_block2(64)
        self.out4 = output_block2(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()  
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up49_woillum(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(5)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block2(64)
        self.out2 = output_block2(64)
        self.out3 = output_block2(64)
        self.out4 = output_block2(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()         

class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up54(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(8)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block5(64)
        self.out2 = output_block5(64)
        self.out3 = output_block5(64)
        self.out4 = output_block5(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x, y,pro1], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()  
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up49_origin(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(3)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block2(64)
        self.out2 = output_block2(64)
        self.out3 = output_block2(64)
        self.out4 = output_block2(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

        x = torch.cat([x], dim=1)
        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()    
        
        
class ResUnet_illum_tran_ds3_add_conv5_sSE_cSE_up49_origin2(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()

        cut = 7

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



        self.conv1x1_5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[2])]
        self.sfs.append(SaveFeatures(base_layers[4][1]))
        self.sfs.append(SaveFeatures(base_layers[5][1]))


        self.up2 = UnetBlock_tran10_bn(256, 128)
        self.up3 = UnetBlock_tran10_bn(128, 64)
        self.up4 = UnetBlock_tran10_bn(64, 64)
        self.ec_block = expand_compress_block3(3)
        self.up5_2 = nn.Conv2d(32, self.num_classes, kernel_size=1, padding=0)
        
        self.out1 = output_block2(64)
        self.out2 = output_block2(64)
        self.out3 = output_block2(64)
        self.out4 = output_block2(64)
        

        self.ds_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.ds_conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.ds_conv5 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.ds_conv6 = nn.Conv2d(64, 32, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(self.num_classes)      
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(32)
        self.bn7 = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):

#        x = torch.cat([x], dim=1)
#        x = self.ec_block(x)
       
        
        x = F.relu(self.rn(x))

        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)

        x_out = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        x_out = self.out4(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = F.upsample(self.sfs[2].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn2(self.ds_conv1(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = F.relu(self.bn3(self.ds_conv2(ds1)))
        ds1 = F.upsample(ds1, scale_factor=2, mode='bilinear', align_corners=True)
        ds1 = self.out1(ds1)

        ds2 = F.upsample(self.sfs[1].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = F.relu(self.bn5(self.ds_conv4(ds2)))
        ds2 = F.upsample(ds2, scale_factor=2, mode='bilinear', align_corners=True)
        ds2 = self.out2(ds2)
        
        ds3 = F.upsample(self.sfs[0].features, scale_factor=2, mode='bilinear', align_corners=True)
        ds3 = self.out3(ds3)

 
        return output,ds1,ds2,ds3

    def close(self):
        for sf in self.sfs: sf.remove()                