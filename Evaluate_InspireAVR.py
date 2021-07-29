
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import natsort
import pandas as pd

from skimage import morphology
from sklearn import metrics
from Tools.BGR2RGB import BGR2RGB
from Tools.BinaryPostProcessing import binaryPostProcessing3
#########################################
def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()
#########################################

ImgNumber = 0
folder = './data/INSPIRE_AVR/Image/'
DF_disc = pd.read_excel(r'D:\项目\RBVS\data\INSPIRE_AVR\DiskParameters_INSPIRE.xls', sheet_name=0)


ImgList = os.listdir(folder)
ImgList = [x for x in ImgList if x.__contains__('.jpg')]
ImgList = natsort.natsorted(ImgList)

pred_folder = './data/INSPIRE_AVR/ResizedLabel_400/'
label_folder = './data/INSPIRE_AVR/ResizedLabel_400/'


predList = os.listdir(pred_folder)
predList = [x for x in predList if x.__contains__('.tif')]
predList = natsort.natsorted(predList)

labelList = os.listdir(label_folder)
labelList = [x for x in labelList if x.__contains__('.tif')]
labelList = natsort.natsorted(labelList)

###############################################

senList = []
specList = []
accList = []

senList_sk = []
specList_sk = []
accList_sk = []

ImgNumber = 0
# for ImgNumber in range(len(ImgList)):
Img = cv2.imread(folder + ImgList[ImgNumber])
Img=cv2.resize(Img,(400,342))
height, width = Img.shape[:2]

ProbMap = cv2.imread(pred_folder+predList[ImgNumber])
Label = cv2.imread(label_folder+labelList[ImgNumber])
ProbMap = BGR2RGB(ProbMap)
Label = BGR2RGB(Label)


ProbMap = np.float32(ProbMap/255.)
Label = np.float32(Label/255.)


discCenter = (DF_disc.loc[ImgNumber, 'DiskCenterRow'], DF_disc.loc[ImgNumber, 'DiskCenterCol'])
discRadius = DF_disc.loc[ImgNumber, 'DiskRadius']
MaskDisc = np.ones((height, width), np.uint8)
cv2.circle(MaskDisc, center=(discCenter[1], discCenter[0]), radius= discRadius, color=0, thickness=-1)

# downsizeRatio = 1000 / (np.maximum(height, width))
#downsizeRatio = 400. / (np.maximum(height, width))  ##TODO: set this to 400 if use the 400 size image; 600 if using 600 size image
#ProbMap = cv2.resize(ProbMap, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
#Label = cv2.resize(Label, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
#MaskDisc = cv2.resize(MaskDisc, dsize=None, fx=downsizeRatio, fy=downsizeRatio)
height, width = Label.shape[:2]


# VesselProb = ProbMap[ImgNumber, 2,:,:]
# VesselLabel = Label[ImgNumber, 2, :, :]
#
# ArteryLabel = Label[ImgNumber, 0, :, :]
# VeinLabel = Label[ImgNumber, 1, :, :]
#
# ArteryProb = ProbMap[ImgNumber, 0,:,:]
# VeinProb = ProbMap[ImgNumber, 1,:,:]
ProbMap = np.load('D:/项目/RBVS/data/ProMap_INSPIRE.npy')
ProbMap = ProbMap[0]

ArteryProb = ProbMap[0,:,:]
VeinProb = ProbMap[2,:,:]
VesselProb = ProbMap[1,:,:]

ArteryLabel = Label[:,:,0] > 0
VeinLabel = Label[:,:,2] > 0
GreenLabel = Label[:,:,1] > 0
ArteryLabel[GreenLabel>0] = 1
VeinLabel[GreenLabel>0] = 1
VesselLabel = np.bitwise_or(ArteryLabel>0, VeinLabel>0)

ArteryLabel = morphology.binary_dilation(ArteryLabel, morphology.disk(5))
VeinLabel = morphology.binary_dilation(VeinLabel, morphology.disk(5))


VesselProb[MaskDisc == 0] = 0
VesselLabel[MaskDisc == 0] = 0
ArteryLabel[MaskDisc == 0] = 0
VeinLabel[MaskDisc == 0] = 0
ArteryProb[MaskDisc == 0] = 0
VeinProb[MaskDisc == 0] = 0

#########################################################
"""Only measure the AV classificaiton metrics on the segmented vessels, while the not segmented ones are not counted"""

VesselSeg = VesselProb >= 0.5
VesselSeg= binaryPostProcessing3(VesselSeg, removeArea=100, fillArea=20)

vesselPixels = np.where(VesselSeg>0)

ArteryProb2 = np.zeros((height,width))
VeinProb2 = np.zeros((height,width))
for i in range(len(vesselPixels[0])):
    row = vesselPixels[0][i]
    col = vesselPixels[1][i]
    probA = ArteryProb[row, col]
    probV = VeinProb[row, col]
#    softmaxProb = softmax(np.array([probA, probV]))
#    ArteryProb2[row, col] = softmaxProb[0]
#    VeinProb2[row, col] = softmaxProb[1]
    ArteryProb2[row, col] = probA
    VeinProb2[row, col] = probV

ArteryLabelImg2= ArteryLabel.copy()
VeinLabelImg2= VeinLabel.copy()
ArteryLabelImg2 [VesselSeg == 0] = 0
VeinLabelImg2 [VesselSeg == 0] = 0
ArteryVeinLabelImg = np.zeros((height, width,3), np.uint8)
ArteryVeinLabelImg[ArteryLabelImg2>0] = (255, 0, 0)
ArteryVeinLabelImg[VeinLabelImg2>0] = (0, 0, 255)
ArteryVeinLabelCommon = np.bitwise_and(ArteryLabelImg2>0, VeinLabelImg2>0)



ArteryPred2 = ArteryProb2 >= 0.5
VeinPred2 = VeinProb2 >= 0.5

# ArteryPred2 = ArteryProb > 0.5
# VeinPred2 = VeinProb > 0.5
ArteryPred2= binaryPostProcessing3(ArteryPred2, removeArea=100, fillArea=20)
VeinPred2= binaryPostProcessing3(VeinPred2, removeArea=100, fillArea=20)


##################################################################################################
##################################################################################################

"""Skeleton Performance Measurement"""
Skeleton = np.uint8(morphology.skeletonize(VesselSeg))
# ArterySkeletonLabel = cv2.bitwise_and(ArteryLabelImg2, ArteryLabelImg2, mask=Skeleton)
# VeinSkeletonLabel = cv2.bitwise_and(VeinLabelImg2, VeinLabelImg2, mask=Skeleton)

ArterySkeletonLabel = ArteryLabelImg2.copy()
ArterySkeletonLabel[Skeleton==0] = 0
VeinSkeletonLabel = VeinLabelImg2.copy()
VeinSkeletonLabel[Skeleton==0] = 0


# ArterySkeletonPred = cv2.bitwise_and(ArteryPred2, ArteryPred2, mask=Skeleton)
# VeinSkeletonPred = cv2.bitwise_and(VeinPred2, VeinPred2, mask=Skeleton)

ArterySkeletonPred = ArteryPred2.copy()
ArterySkeletonPred[Skeleton == 0] = 0
VeinSkeletonPred = VeinPred2.copy()
VeinSkeletonPred[Skeleton == 0] = 0



ArteryVeinPred_sk = np.zeros((height, width, 3), np.uint8)
skeletonPixles = np.where(Skeleton > 0)

TPa_sk = 0
TNa_sk = 0
FPa_sk = 0
FNa_sk = 0
for i in range(len(skeletonPixles[0])):
    row = skeletonPixles[0][i]
    col = skeletonPixles[1][i]
    if ArterySkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1:
        TPa_sk = TPa_sk + 1
        ArteryVeinPred_sk[row, col] = (255, 0, 0)
    elif VeinSkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1:
        TNa_sk = TNa_sk + 1
        ArteryVeinPred_sk[row, col] = (0, 0, 255)
    elif ArterySkeletonLabel[row, col] == 1 and VeinSkeletonPred[row, col] == 1 \
            and ArteryVeinLabelCommon[row, col] == 0:
        FNa_sk = FNa_sk + 1
        ArteryVeinPred_sk[row, col] = (255, 255, 0)
    elif VeinSkeletonLabel[row, col] == 1 and ArterySkeletonPred[row, col] == 1 \
            and ArteryVeinLabelCommon[row, col] == 0:
        FPa_sk = FPa_sk + 1
        ArteryVeinPred_sk[row, col] = (0, 255, 255)
    else:
        pass

sensitivity_sk = TPa_sk / (TPa_sk + FNa_sk)
specificity_sk = TNa_sk / (TNa_sk + FPa_sk)
acc_sk = (TPa_sk + TNa_sk) / (TPa_sk + TNa_sk + FPa_sk + FNa_sk)

senList_sk.append(sensitivity_sk)
specList_sk.append(specificity_sk)
accList_sk.append(acc_sk)
print('Skeletonal Metrics', acc_sk, sensitivity_sk, specificity_sk)


# print('Avg Pixel-wise Performance:', np.mean(accList), np.mean(senList), np.mean(specList))
print('Avg Skeleton Performance:', np.mean(accList_sk), np.mean(senList_sk), np.mean(specList_sk))



######################################################
##################################################################################################
print("End of Image Processing >>>")
plt.figure()
Images = [BGR2RGB(Img), ArteryVeinLabelImg, ArteryProb2,
           ArteryPred2,  VeinPred2, ArteryVeinPred_sk]
Titles = [ 'ImgShow', 'ArteryVeinLabelImg',   'ArteryProb2',
            'ArteryPred2',  'VeinPred2', 'ArteryVeinPred_sk',]

cv2.imwrite('./ArteryVeinLabelImg.png',ArteryVeinLabelImg)


for i in range(0, len(Images)):
    plt.subplot(2, 4, i + 1), plt.imshow(Images[i], 'gray'), plt.title(Titles[i])
plt.show()

