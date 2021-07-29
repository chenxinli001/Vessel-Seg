from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50
import torch.nn.functional as F
from models.unet_parts import *
from models.layers import *


class ResUnet(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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


        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock(128, 64, 256)
        self.up3 = UnetBlock(256, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = self.up3(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        return output

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock(128, 64, 256)
        self.up3 = UnetBlock(256, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x,y):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = self.up3(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        return output

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(256, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x,y):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = self.up3(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        return output

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_trad_conv3_ds(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(256, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_ds = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = self.up3(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn_ds(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn_ds(ds2)

        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_trad_conv_ds(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(288, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(160,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(160, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(32)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_ds = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        
        pro1 = self.conv1x1_3(pro1)
        pro1 = F.relu(self.bn(pro1))
        pro2 = self.conv1x1_3(pro2)
        pro2 = F.relu(self.bn(pro2))
        pro3 = self.conv1x1_3(pro3)
        pro3 = F.relu(self.bn(pro3))

        
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = torch.cat([x, pro3], dim=1)
        x = self.up3(x, self.sfs[0].features)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn_ds(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn_ds(ds2)

        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()       

class ResUnet_illum_tran_trad_conv_ds2(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        self.conv1x1_4 = nn.Conv2d(3, 2, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(288, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(160,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(160, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(32)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn_ds = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        
        pro1 = self.conv1x1_3(pro1)
        pro1 = F.relu(self.bn(pro1))
        pro2 = self.conv1x1_3(pro2)
        pro2 = F.relu(self.bn(pro2))
        pro3 = self.conv1x1_3(pro3)
        pro3 = F.relu(self.bn(pro3))

        
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = torch.cat([x, pro3], dim=1)
        x = self.up3(x, self.sfs[0].features)
        x = torch.cat([x, pro2], dim=1)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = torch.cat([x_out, pro1], dim=1)
        x_out = self.up5_2(x_out)


#        if self.num_classes==1:
#            output = x_out[:, 0]
#        else:
#            output = x_out[:, :self.num_classes]
        
        output = x_out[:,0].unsqueeze(1)
        x_out = self.conv1x1_4(x_out)
        output = torch.cat([x_out, output], dim=1)

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn_ds(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn_ds(ds2)

        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()      

class ResUnet_illum_tran_ds(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
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
        self.conv1x1_2 = nn.Conv2d(6, 3, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(256, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = self.up3(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)

        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds2(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        self.conv1x1_3 = nn.Conv2d(3, 2, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(256, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = self.up3(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


#        if self.num_classes==1:
#            output = x_out[:, 0]
#        else:
#            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)

        output = x_out[:,0].unsqueeze(1)
        x_out = self.conv1x1_3(x_out)
        output = torch.cat([x_out, output], dim=1)

        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds2_add(nn.Module):

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
        self.conv1x1_3 = nn.Conv2d(3, 2, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(256, 128,256)
        self.up3 = UnetBlock_tran(256, 64,256)
        self.up4 = UnetBlock_tran(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


#        if self.num_classes==1:
#            output = x_out[:, 0]
#        else:
#            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)

        output = x_out[:,0].unsqueeze(1)
        x_out = self.conv1x1_3(x_out)
        output = torch.cat([x_out, output], dim=1)

        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()


class ResUnet_illum_tran_ds_add(nn.Module):

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
        self.conv1x1_3 = nn.Conv2d(3, 2, kernel_size=1, padding=0)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(256, 128,256)
        self.up3 = UnetBlock_tran(256, 64,256)
        self.up4 = UnetBlock_tran(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds_add_conv(nn.Module):

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
        self.up2 = UnetBlock_tran(256, 128,256)
        self.up3 = UnetBlock_tran(256, 64,256)
        self.up4 = UnetBlock_tran(288, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(160,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(160, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()


class ResUnet_illum_tran_ds_add_conv3(nn.Module):

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
        self.up2 = UnetBlock_tran(256, 128,256)
        self.up3 = UnetBlock_tran(256, 64,256)
        self.up4 = UnetBlock_tran(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()
        
class ResUnet_illum_tran_ds_add_conv4(nn.Module):

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
        self.up2 = UnetBlock_tran6(256, 128,256)
        self.up3 = UnetBlock_tran6(256, 64,256)
        self.up4 = UnetBlock_tran6(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds_add_conv3_unet(nn.Module):

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
        self.up2 = UnetBlock_tran4(256, 128,256)
        self.up3 = UnetBlock_tran4(256, 64,256)
        self.up4 = UnetBlock_tran4(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds_add_conv3_sSE_cSE(nn.Module):

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
        self.up2 = UnetBlock_tran_scSE(256, 128,256)
        self.up3 = UnetBlock_tran_scSE(256, 64,256)
        self.up4 = UnetBlock_tran_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()
        
class ResUnet_illum_tran_ds_add_conv4_sSE_cSE(nn.Module):

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
        self.up2 = UnetBlock_tran6_scSE(256, 128,256)
        self.up3 = UnetBlock_tran6_scSE(256, 64,256)
        self.up4 = UnetBlock_tran6_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds_add_conv4_sSE_cSE_up(nn.Module):

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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds_add_conv5_sSE_cSE_up(nn.Module):

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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds_add_conv5_sSE_cSE(nn.Module):

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
        self.up2 = UnetBlock_tran9_scSE(256, 128,256)
        self.up3 = UnetBlock_tran9_scSE(256, 64,256)
        self.up4 = UnetBlock_tran9_scSE(256, 64,128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)


        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_ds3_add_conv(nn.Module):

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
        self.up2 = UnetBlock_tran(256, 128,256)
        self.up3 = UnetBlock_tran(256, 64,256)
        self.up4 = UnetBlock_tran(288, 64,128)

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

class UNet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2):
        super(UNet, self).__init__()
        self.inc = inconv(6, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, num_classes)

    def forward(self, x,y):
        x = torch.cat([x, y], dim=1)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

class UNet_ds(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2):
        super(UNet_ds, self).__init__()
        self.conv1 = double_conv(6, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.conv5 = double_conv(512, 1024)
        self.conv1_up = double_conv(1024, 512)
        self.conv2_up = double_conv(512, 256)
        self.conv3_up = double_conv(256, 128)
        self.conv4_up = double_conv(128, 64)
        self.maxpool = nn.MaxPool2d(2)
        self.deconv1 = nn.ConvTranspose2d(1024,512,kernel_size=2,stride=2)
        self.deconv2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.deconv3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.deconv4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.outc = outconv(64,num_classes)
        
        self.ds_deconv1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_deconv2 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(num_classes)

    def forward(self, x,y):
        x = torch.cat([x, y], dim=1)
        x1 = self.conv1(x)
        x1_dn = self.maxpool(x1)
        x2 = self.conv2(x1_dn)
        x2_dn = self.maxpool(x2)
        x3 = self.conv3(x2_dn)
        x3_dn = self.maxpool(x3)
        x4 = self.conv4(x3_dn)
        x4_dn = self.maxpool(x4)
        x5 = self.conv5(x4_dn)
        
        x = self.deconv1(x5)
        x = torch.cat([x, x4], dim=1)
        x = self.conv1_up(x)
        
        x = self.deconv2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2_up(x)
        
        x = self.deconv3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3_up(x)
        
        x = self.deconv4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4_up(x)
        
        x = self.outc(x)
        
        ds1 = self.ds_deconv1(x1_dn)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv2(x2_dn)
        ds2 = self.ds_deconv1(ds2)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)
        
        return x,ds1,ds2

class ResUnet_illum_tran_ds3_add_conv3(nn.Module):

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
        self.up2 = UnetBlock_tran(256, 128,256)
        self.up3 = UnetBlock_tran(256, 64,256)
        self.up4 = UnetBlock_tran(256, 64,128)

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

class ResUnet_illum_tran_up2_ds(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran2(128, 64)
        self.up3 = UnetBlock_tran2(64, 64)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = self.up3(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)

        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_add_up2_ds(nn.Module):

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
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2,4]]
        self.sfs.append(SaveFeatures(base_layers[5][0]))
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran2(256, 128)
        self.up3 = UnetBlock_tran2(128, 64)
        self.up4 = UnetBlock_tran2(64, 64)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        
        self.ds_deconv = nn.ConvTranspose2d(64,64,kernel_size=2,stride=2)
        self.ds_conv = nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(self.num_classes)

    def forward(self, x,y):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
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

        ds1 = self.ds_deconv(self.sfs[1].features)
        ds1 = self.ds_deconv(ds1)
        ds1 = self.ds_conv(ds1)
        ds1 = self.bn(ds1)
        
        ds2 = self.ds_deconv(self.sfs[0].features)
        ds2 = self.ds_conv(ds2)
        ds2 = self.bn(ds2)

        return output,ds1,ds2

    def close(self):
        for sf in self.sfs: sf.remove()

class ResUnet_illum_tran_trad(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(258, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(130,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(130, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = torch.cat([x, pro3], dim=1)
        x = self.up3(x, self.sfs[0].features)
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

        return output

    def close(self):
        for sf in self.sfs: sf.remove()
        
class ResUnet_illum_tran_trad_conv(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(288, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(160,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(160, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        
        pro1 = self.conv1x1_3(pro1)
        pro1 = F.relu(self.bn(pro1))
        pro2 = self.conv1x1_3(pro2)
        pro2 = F.relu(self.bn(pro2))
        pro3 = self.conv1x1_3(pro3)
        pro3 = F.relu(self.bn(pro3))

        
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = torch.cat([x, pro3], dim=1)
        x = self.up3(x, self.sfs[0].features)
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

        return output

    def close(self):
        for sf in self.sfs: sf.remove()        
   


     
class ResUnet_illum_tran_trad_conv2(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        self.conv1x1_3 = nn.Conv2d(2, 32, kernel_size=3, padding=1)
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(288, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(160,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(160, self.num_classes, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        
        pro1 = self.conv1x1_3(pro1)
        pro1 = F.relu(self.bn(pro1))
        pro2 = self.conv1x1_3(pro2)
        pro2 = F.relu(self.bn(pro2))
        pro3 = self.conv1x1_3(pro3)
        pro3 = F.relu(self.bn(pro3))
        
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = torch.cat([x, pro3], dim=1)
        x = self.up3(x, self.sfs[0].features)
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

        return output

    def close(self):
        for sf in self.sfs: sf.remove()  


class ResUnet_illum_tran_trad_conv3(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2):
        super().__init__()
        # super(ResUnet, self).__init__()
        # cut, lr_cut = [8, 6]
        cut, lr_cut = [6, 5]

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
        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4]]
        #self.up1 = UnetBlock0(512, 256, 256)
        #self.up1 = UnetBlock(256, 128, 256)
        self.up2 = UnetBlock_tran(128, 64, 256)
        self.up3 = UnetBlock_tran(256, 64, 128)

        # self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)
        # self.up5_1 = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.up5_2 = nn.Conv2d(256, self.num_classes, 1)

        #self.up5_1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.up5_1 = nn.ConvTranspose2d(128,128,kernel_size=2,stride=2)
        self.up5_2 = nn.Conv2d(128, self.num_classes, kernel_size=1, padding=0)

    def forward(self, x,y,pro1,pro2,pro3):
        #x = self.conv1(x)
        #y = self.conv1(y)
        x = torch.cat([x, y,pro1], dim=1)
        x = self.conv1x1_2(x)
        #x = self.conv2(x)
        #x = self.conv1x1(x)
        x = F.relu(self.rn(x))
        # x = self.up1(x, self.sfs[2].features)
        x = self.up2(x, self.sfs[1].features)
        x = self.up3(x, self.sfs[0].features)
        # x = self.up4(x, self.sfs[0].features)
        # x_out = self.up5(x)
        x_out = self.up5_1(x)
        x_out = self.up5_2(x_out)


        if self.num_classes==1:
            output = x_out[:, 0]
        else:
            output = x_out[:, :self.num_classes]

        return output

    def close(self):
        for sf in self.sfs: sf.remove()
        
