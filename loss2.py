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

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=2, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs,ds1,ds2,ds3,targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.sigmoid(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1).long()
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

#class FocalLoss2d(nn.Module):
#    def __init__(self, gamma=2, size_average=True):
#        super(FocalLoss2d, self).__init__()
#        self.gamma = gamma
#        self.size_average = size_average
#
#    def forward(self, preds,ds1,ds2,ds3, targs, class_weight=None, type='sigmoid'):
#        targs = targs.view(-1, 1).long()
#        target = targs[:,0,:,:]
#        logit = preds[:,0]
##        target_artery = target[:,0,:,:]
##        target_vein = target[:,1,:,:]
##        target_all = target[:,2,:,:]
#        
#        if type=='sigmoid':
#            if class_weight is None:
#                class_weight = [1]*2 #[0.5, 0.5]
#            prob   = F.sigmoid(logit)
#            prob   = prob.view(-1, 1)
#            prob   = torch.cat((1-prob, prob), 1)
#            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
#            select.scatter_(1, target, 1.)
#        elif  type=='softmax':
#            B,C,H,W = logit.size()
#            if class_weight is None:
#                class_weight =[1]*C #[1/C]*C
#            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
#            prob    = F.softmax(logit,1)
#            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
#            select.scatter_(1, target, 1.)
#        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
#        class_weight = torch.gather(class_weight, 0, target)
#        prob       = (prob*select).sum(1).view(-1,1)
#        prob       = torch.clamp(prob,1e-8,1-1e-8)
#        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()
#        if self.size_average:
#            loss = batch_loss.mean()
#        else:
#            loss = batch_loss
#        return loss		

class FocalLoss2d(nn.Module):

    def __init__(self, gamma=2, weight=None, size_average=True):
        super(FocalLoss2d, self).__init__()

        self.gamma = gamma
        self.weight = weight
        self.size_average = size_average

    def forward(self, preds,ds1,ds2,ds3,targs):
        preds = F.sigmoid(preds)
        ds1 = F.sigmoid(ds1)
        ds2 = F.sigmoid(ds2)
        ds3 = F.sigmoid(ds3)
        input_a = preds[:,0,:,:]
        input_v = preds[:,2,:,:]
        input_all = preds[:,1,:,:]
        
        input_a1 = ds1[:,0,:,:]
        input_v1 = ds1[:,2,:,:]
        input_all1 = ds1[:,1,:,:]
        input_a2 = ds2[:,0,:,:]
        input_v2 = ds2[:,2,:,:]
        input_all2 = ds2[:,1,:,:]
        input_a3 = ds3[:,0,:,:]
        input_v3 = ds3[:,2,:,:]
        input_all3 = ds3[:,1,:,:]
        
        target_artery = targs[:,0,:,:]
        target_vein = targs[:,1,:,:]
        target_all = targs[:,2,:,:]

#            input = input.contiguous().view(input.size(0), input.size(1), -1)
#            input = input.transpose(1,2)
#            input = input.contiguous().view(-1, input.size(2)).squeeze()
        input_a = input_a.contiguous().view(-1)
        input_v = input_v.contiguous().view(-1)
        input_all = input_all.contiguous().view(-1)
        input_a1 = input_a1.contiguous().view(-1)
        input_v1 = input_v1.contiguous().view(-1)
        input_all1 = input_all1.contiguous().view(-1)
        input_a2 = input_a2.contiguous().view(-1)
        input_v2 = input_v2.contiguous().view(-1)
        input_all2 = input_all2.contiguous().view(-1)
        input_a3 = input_a3.contiguous().view(-1)
        input_v3 = input_v3.contiguous().view(-1)
        input_all3 = input_all3.contiguous().view(-1)
        #if target.dim()==4:
#            target = target.contiguous().view(target.size(0), target.size(1), -1)
#            target = target.transpose(1,2)
#            target = target.contiguous().view(-1, target.size(2)).squeeze()
        
        target_artery = target_artery.contiguous().view(-1)
        target_vein = target_vein.contiguous().view(-1)
        target_all = target_all.contiguous().view(-1)
        #if:
#            target = target.contiguous().view(-1, 1)

        # compute the negative likelyhood
        arteryWeight = 2
        veinWeight = 2
        vesselWeight = 3
                
        #input = F.sigmoid(input)
        logpt_a = -F.binary_cross_entropy(input_a, target_artery)
        logpt_v = -F.binary_cross_entropy(input_v, target_vein)
        logpt_all = -F.binary_cross_entropy(input_all, target_all)       
        pt_a = torch.exp(logpt_a)
        pt_v = torch.exp(logpt_v)
        pt_all = torch.exp(logpt_all)
        # compute the loss
        loss_a = -((1-pt_a)**self.gamma) * logpt_a
        loss_v = -((1-pt_v)**self.gamma) * logpt_v
        loss_all = -((1-pt_all)**self.gamma) * logpt_all

        logpt_a1 = -F.binary_cross_entropy(input_a1, target_artery)
        logpt_v1 = -F.binary_cross_entropy(input_v1, target_vein)
        logpt_all1 = -F.binary_cross_entropy(input_all1, target_all)       
        pt_a1 = torch.exp(logpt_a1)
        pt_v1 = torch.exp(logpt_v1)
        pt_all1 = torch.exp(logpt_all1)
        # compute the loss
        loss_a1 = -((1-pt_a1)**self.gamma) * logpt_a1
        loss_v1 = -((1-pt_v1)**self.gamma) * logpt_v1
        loss_all1 = -((1-pt_all1)**self.gamma) * logpt_all1
        
        logpt_a2 = -F.binary_cross_entropy(input_a2, target_artery)
        logpt_v2 = -F.binary_cross_entropy(input_v2, target_vein)
        logpt_all2 = -F.binary_cross_entropy(input_all2, target_all)       
        pt_a2 = torch.exp(logpt_a2)
        pt_v2 = torch.exp(logpt_v2)
        pt_all2 = torch.exp(logpt_all2)
        # compute the loss
        loss_a2 = -((1-pt_a2)**self.gamma) * logpt_a2
        loss_v2 = -((1-pt_v2)**self.gamma) * logpt_v2
        loss_all2 = -((1-pt_all2)**self.gamma) * logpt_all2
        
        logpt_a3 = -F.binary_cross_entropy(input_a3, target_artery)
        logpt_v3 = -F.binary_cross_entropy(input_v3, target_vein)
        logpt_all3 = -F.binary_cross_entropy(input_all3, target_all)       
        pt_a3 = torch.exp(logpt_a3)
        pt_v3 = torch.exp(logpt_v3)
        pt_all3 = torch.exp(logpt_all3)
        # compute the loss
        loss_a3 = -((1-pt_a3)**self.gamma) * logpt_a3
        loss_v3 = -((1-pt_v3)**self.gamma) * logpt_v3
        loss_all3 = -((1-pt_all3)**self.gamma) * logpt_all3


        loss = (arteryWeight*loss_a +
                 veinWeight*loss_v +
                 vesselWeight*loss_all) / (arteryWeight+veinWeight+vesselWeight) + 1/3.0*(arteryWeight*loss_a1 +
                 veinWeight*loss_v1 +
                 vesselWeight*loss_all1) / (arteryWeight+veinWeight+vesselWeight) + 1/3.0*(arteryWeight*loss_a2 +
                 veinWeight*loss_v2 +
                 vesselWeight*loss_all2) / (arteryWeight+veinWeight+vesselWeight) + 1/3.0*(arteryWeight*loss_a3 +
                 veinWeight*loss_v3 +
                 vesselWeight*loss_all3) / (arteryWeight+veinWeight+vesselWeight)

        # averaging (or not) loss
#        if self.size_average:
#            return loss.mean()
#        else:
#            return loss.sum()		
        return 100*loss
     
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

        loss = (arteryWeight*self.logitsLoss(preds[:,0], target_artery) +
                 veinWeight*self.logitsLoss(preds[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(preds[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + 1/3.0*(arteryWeight*self.logitsLoss(ds1[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds1[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds1[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + 1/3.0*(arteryWeight*self.logitsLoss(ds2[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds2[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds2[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + 1/3.0*(arteryWeight*self.logitsLoss(ds3[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds3[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds3[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight)
                   
        return loss
    
    
class multiclassLoss_ds4():
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
        vesselWeight = 0

        loss = 2*(arteryWeight*self.logitsLoss(preds[:,0], target_artery) +
                 veinWeight*self.logitsLoss(preds[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(preds[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight)
                   
        return loss   
    
class multiclassLoss_ds5():
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

        loss = (arteryWeight*self.logitsLoss(preds[:,0], target_artery) +
                 veinWeight*self.logitsLoss(preds[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(preds[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + 1/3.0*(
                 veinWeight*self.logitsLoss(ds1[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds1[:,1], target_all)) / (veinWeight+vesselWeight) + 1/3.0*(arteryWeight*self.logitsLoss(ds2[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds2[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds2[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight) + 1/3.0*(arteryWeight*self.logitsLoss(ds3[:,0], target_artery) +
                 veinWeight*self.logitsLoss(ds3[:,2], target_vein) +
                 vesselWeight*self.logitsLoss(ds3[:,1], target_all)) / (arteryWeight+veinWeight+vesselWeight)
                   
        return loss    
