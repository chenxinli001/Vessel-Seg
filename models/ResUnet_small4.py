from torch import nn
import torch
from models.resnet import resnet34, resnet18, resnet50
import torch.nn.functional as F


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock0(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        # self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=1, padding=0)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))

class UnetBlock_tran(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, kernel_size=3, padding=1)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.upsample = nn.ConvTranspose2d(up_in,up_in,kernel_size=2,stride=2)
        self.conv1x1 = nn.Conv2d(up_in, up_out, kernel_size=1, padding=0)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):

        up_p = self.upsample(up_p)
        up_p = self.conv1x1(up_p)

        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


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
