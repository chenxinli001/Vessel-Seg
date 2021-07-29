import torch.nn.functional as F
from torch import nn
import torch 
import numpy
from torch.autograd import Variable


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)
#    
#_inputs = [[[[1,2,3],[2,3,4]],[[0.1,2,3],[2,6,4]]]]
#label = [[[0,0,1],[0,0,0]]]
#_inputs = torch.from_numpy(numpy.array(_inputs).astype(numpy.float))
#label = torch.from_numpy(numpy.array(label))
#loss = CrossEntropyLoss2d()
#a = Variable(_inputs)
#b = Variable(label)
#print(_inputs,label)
#print(F.softmax(a))
#print(loss(a,b).data[0])
        
class multiclassLoss():
    def __init__(self,num_classes=3):
        self.num_classes = num_classes
        self.logitsLoss = nn.BCEWithLogitsLoss()
        
    def __call__(self, preds, targs):
        #print(preds.shape, targs.shape)
        
#        target_artery = (targs == 2).float()
#        target_vein = (targs == 1).float()
#        target_all = (targs >= 1).float()
        target_artery = targs[:,0,:,:]
        target_vein = targs[:,1,:,:]
        target_all = targs[:,2,:,:]
        arteryWeight = 2
        veinWeight = 2
        vesselWeight = 3        

        loss = ( arteryWeight*self.logitsLoss(preds[:,0], target_artery) + 
                 veinWeight*self.logitsLoss(preds[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(preds[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight)
        return loss

class multiclassLoss_ds():
    def __init__(self,num_classes=3):
        self.num_classes = num_classes
        self.logitsLoss = nn.BCEWithLogitsLoss()

    def __call__(self, preds,ds1,ds2, targs):
        #print(preds.shape, targs.shape)

#        target_artery = (targs == 2).float()
#        target_vein = (targs == 1).float()
#        target_all = (targs >= 1).float()
        target_artery = targs[:,0,:,:]
        target_vein = targs[:,1,:,:]
        target_all = targs[:,2,:,:]
        arteryWeight = 2
        veinWeight = 2
        vesselWeight = 3

        loss = (arteryWeight*self.logitsLoss(preds[:,0], target_artery) +
                 veinWeight*self.logitsLoss(preds[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(preds[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + 0.4*(arteryWeight*self.logitsLoss(ds1[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds1[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds1[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + 0.3*(arteryWeight*self.logitsLoss(ds2[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds2[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds2[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight)
                   
        return loss

class multiclassLoss_ds3():
    def __init__(self,num_classes=3):
        self.num_classes = num_classes
        self.logitsLoss = nn.BCEWithLogitsLoss()

    def __call__(self, preds,ds1,ds2,ds3,targs):
        #print(preds.shape, targs.shape)

#        target_artery = (targs == 2).float()
#        target_vein = (targs == 1).float()
#        target_all = (targs >= 1).float()
        target_artery = targs[:,0,:,:]
        target_vein = targs[:,1,:,:]
        target_all = targs[:,2,:,:]
        arteryWeight = 2
        veinWeight = 2
        vesselWeight = 3
        end_weight = 1
        ds_weight1 = 0.4
        ds_weight2 = 0.3
        ds_weight3 = 0.2


        loss = end_weight*(arteryWeight*self.logitsLoss(preds[:,0], target_artery) +
                 veinWeight*self.logitsLoss(preds[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(preds[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + ds_weight1*(arteryWeight*self.logitsLoss(ds1[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds1[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds1[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + ds_weight2*(arteryWeight*self.logitsLoss(ds2[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds2[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds2[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + ds_weight3*(arteryWeight*self.logitsLoss(ds3[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds3[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds3[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight)
                   
        return loss
