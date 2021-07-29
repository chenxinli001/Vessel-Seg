
import os
import h5py
import numpy as np
from PIL import Image
import cv2
import natsort
from lib.IlluminationCorrection import illuminationCorrection2
from lib.extract_patches import *


def write_hdf5(arr,outfile):
    with h5py.File(outfile,"w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


#------------Path of the images --------------------------------------------------------------
#train
save_path = r"./data/AV_DRIVE/DL_Data/"


folder_imgs_train = r"./data/AV_DRIVE/training/images/"
folder_avLabel_train = r"./data/AV_DRIVE/training/av/"
folder_borderMasks_train = r"./data/AV_DRIVE/training/mask/"
#test
folder_imgs_test = r"./data/AV_DRIVE/test\images/"
folder_avLabel_test = r"./data/AV_DRIVE/test/av/"
folder_borderMasks_test = r"./data/AV_DRIVE/test/mask/"

ImgList0 = os.listdir(folder_imgs_train)
ImgList0 = natsort.natsorted(ImgList0)
ImgList = [x for x in ImgList0 if x.__contains__('.tif')]



Nimgs = 20
channels = 3
height = 584
width = 565


patch_h = 64
patch_w = 64
patch_per_img = 4000
img_w = 565
img_h = 565
inside_FOV = True



iter_tot = 0
for i, imgName in enumerate(ImgList):
    label_name = imgName[0:2] + "_training.png"
    FGmask_name = imgName[0:2] + "_training_mask.jpg"
    FG_Mask = cv2.imread(folder_borderMasks_train + FGmask_name, 0)
    LabelImg = cv2.imread(folder_avLabel_train + label_name)

    Img = cv2.imread(folder_imgs_train + imgName)
    IllumImg = illuminationCorrection2(Img, kernel_size=35)


    """encode the ground truth into 3 channels:
    channel1: Blue Vein
    Channel3: Red Artery
    Channel2: all vessels"""
    WhiteMap = np.bitwise_and(LabelImg[:, :, 0] > 0, LabelImg[:, :, 1] > 0)
    WhiteMap = np.bitwise_and(WhiteMap, LabelImg[:, :, 2] > 0)
    GreenMap = np.bitwise_and(LabelImg[:, :, 1] > 0, np.bitwise_not(WhiteMap))

    BlueMap = np.bitwise_and(LabelImg[:, :, 0] > 0, np.bitwise_not(WhiteMap))
    BlueMap = np.bitwise_or(BlueMap, GreenMap)

    RedMap = np.bitwise_and(LabelImg[:, :, 2] > 0, np.bitwise_not(WhiteMap))
    RedMap = np.bitwise_or(RedMap, GreenMap)

    AllMap = np.bitwise_or(LabelImg[:, :, 0] > 0, LabelImg[:, :, 1] > 0)
    AllMap = np.bitwise_or(AllMap, LabelImg[:, :, 2] > 0)

    EncodedLabelImg = np.zeros(LabelImg.shape[:2], np.uint8)
    # EncodedLabelImg[BlueMap, 0] = 255
    # EncodedLabelImg[RedMap, 2] = 255
    # EncodedLabelImg[AllMap, 1] = 255
    EncodedLabelImg[RedMap] = 255
    EncodedLabelImg[BlueMap] = 150



    #######################
    """Extract Patches"""
    train_img = Img[9:574, :, :]
    train_label = EncodedLabelImg[9:574, :]

    # ImgPatchs = np.empty((patch_per_img, patch_h, patch_w, 3))
    # LabelPatchs = np.empty((patch_per_img, patch_h, patch_w, 3))

    k = 0
    while k < patch_per_img:
        x_center = random.randint(0 + int(patch_w / 2), img_w - int(patch_w / 2))
        # print "x_center " +str(x_center)
        y_center = random.randint(0 + int(patch_h / 2), img_h - int(patch_h / 2))
        # print "y_center " +str(y_center)
        # check whether the patch is fully contained in the FOV
        if inside_FOV == True:
            if is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h) == False:
                continue
        patch = train_img[y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                x_center - int(patch_w / 2):x_center + int(patch_w / 2), : ]
        patch_label = train_label[ y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                     x_center - int(patch_w / 2):x_center + int(patch_w / 2)]

        # ImgPatchs[k, :,:,:] = patch
        # LabelPatchs[k, :, :, :] = patch_label
        iter_tot += 1  # total
        k += 1  # per full_img

        """save as imgs"""
        if not os.path.exists(save_path + 'ImgPatches_Img\\' + imgName[0:2] + "_train\\"):
            os.mkdir(save_path + 'ImgPatches_Img\\' + imgName[0:2] + "_train\\")
        if not os.path.exists(save_path + 'LabelPatches_Img\\' + imgName[0:2] + "_train\\"):
            os.mkdir(save_path + 'LabelPatches_Img\\' + imgName[0:2] + "_train\\")

        save_name_img = save_path + 'ImgPatches_Img\\' + imgName[0:2] + "_train\\" + str(k) + '.png'
        save_name_label = save_path + 'LabelPatches_Img\\' + imgName[0:2] + "_train\\" + str(k) + '.png'
        cv2.imwrite(save_name_img, patch)
        cv2.imwrite(save_name_label, patch_label)

    print(imgName, k)




    """Save as hdf5 file"""
    # save_name_img = save_path + 'ImgPatches\\' + imgName[0:2] + "_train.hdf5"
    # save_name_label = save_path + 'LabelPatches\\' + imgName[0:2] + "_train.hdf5"
    # write_hdf5(ImgPatchs,save_name_img)
    # write_hdf5(LabelPatchs,save_name_label)


    # print('saving all patches')

# write_hdf5(ImgPatchs,  save_path + "ImgPatchs_DRIVE_Train.hdf5")
# write_hdf5(LabelPatchs,  save_path + "LabelPatchs_DRIVE_Train.hdf5")





