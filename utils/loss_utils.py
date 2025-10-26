import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from LovaszSoftmax.pytorch import lovasz_losses as L


class LovaszLossSoftmax(nn.Module):
    def __init__(self):
        super(LovaszLossSoftmax, self).__init__()

    def forward(self, input, target):

        input = input.to(dtype=torch.float32)
        target = target.to(dtype=torch.float32)
        out = F.softmax(input, dim=1)
        loss = L.lovasz_softmax(out, target)
        return loss



def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat([Y, Cr, Cb], dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
        .transpose(1, 3)
        .transpose(2, 3)
    )
    return out


def Sobelxy(x):
    kernelx = [[-1, 0, 1],
              [-2,0 , 2],
              [-1, 0, 1]]
    kernely = [[1, 2, 1],
              [0,0 , 0],
              [-1, -2, -1]]
    kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
    kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
    weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
    weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    
    sobelx=F.conv2d(x, weightx, padding=1)
    sobely=F.conv2d(x, weighty, padding=1)
    return sobelx, sobely


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.l1_loss =  nn.L1Loss()

    def forward(self, input_xy, output, Mask):
        input_vis, input_ir = input_xy 

        Fuse = output * Mask
        YCbCr_Fuse = RGB2YCrCb(Fuse) 
        Y_Fuse  = YCbCr_Fuse[:,0:1,:,:]
        Cr_Fuse = YCbCr_Fuse[:,1:2,:,:]
        Cb_Fuse = YCbCr_Fuse[:,2:,:,:]  
        

        R_vis = torchvision.transforms.functional.adjust_gamma(input_vis, 0.8, 1)
        YCbCr_R_vis = RGB2YCrCb(R_vis) 
        Y_R_vis = YCbCr_R_vis[:,0:1,:,:]
        Cr_R_vis = YCbCr_R_vis[:,1:2,:,:]
        Cb_R_vis = YCbCr_R_vis[:,2:,:,:]          
        
                        
        R_ir = torchvision.transforms.functional.adjust_contrast(input_ir, 1)


        Fuse_R = torch.unsqueeze(Fuse[:,0,:,:],1)
        Fuse_G = torch.unsqueeze(Fuse[:,1,:,:],1)
        Fuse_B = torch.unsqueeze(Fuse[:,2,:,:],1)
        Fuse_R_grad_x,Fuse_R_grad_y =   Sobelxy(Fuse_R)
        Fuse_G_grad_x,Fuse_G_grad_y =   Sobelxy(Fuse_G)
        Fuse_B_grad_x,Fuse_B_grad_y =   Sobelxy(Fuse_B)
        Fuse_grad_x = torch.cat([Fuse_R_grad_x, Fuse_G_grad_x, Fuse_B_grad_x], 1)
        Fuse_grad_y = torch.cat([Fuse_R_grad_y, Fuse_G_grad_y, Fuse_B_grad_y], 1)


        R_VIS_R = torch.unsqueeze(R_vis[:,0,:,:],1)
        R_VIS_G = torch.unsqueeze(R_vis[:,1,:,:],1)
        R_VIS_B = torch.unsqueeze(R_vis[:,2,:,:],1)
        R_VIS_R_grad_x, R_VIS_R_grad_y =   Sobelxy(R_VIS_R)
        R_VIS_G_grad_x, R_VIS_G_grad_y =   Sobelxy(R_VIS_G)
        R_VIS_B_grad_x, R_VIS_B_grad_y =   Sobelxy(R_VIS_B)
        R_VIS_grad_x = torch.cat([R_VIS_R_grad_x, R_VIS_G_grad_x, R_VIS_B_grad_x], 1)
        R_VIS_grad_y = torch.cat([R_VIS_R_grad_y, R_VIS_G_grad_y, R_VIS_B_grad_y], 1)


        R_IR_R = torch.unsqueeze(R_ir[:,0,:,:],1)
        R_IR_G = torch.unsqueeze(R_ir[:,1,:,:],1)
        R_IR_B = torch.unsqueeze(R_ir[:,2,:,:],1)
        R_IR_R_grad_x,R_IR_R_grad_y =   Sobelxy(R_IR_R)
        R_IR_G_grad_x,R_IR_G_grad_y =   Sobelxy(R_IR_G)
        R_IR_B_grad_x,R_IR_B_grad_y =   Sobelxy(R_IR_B)
        R_IR_grad_x = torch.cat([R_IR_R_grad_x, R_IR_G_grad_x,R_IR_B_grad_x], 1)
        R_IR_grad_y = torch.cat([R_IR_R_grad_y, R_IR_G_grad_y,R_IR_B_grad_y], 1)


        joint_grad_x = torch.maximum(R_VIS_grad_x, R_IR_grad_x)
        joint_grad_y = torch.maximum(R_VIS_grad_y, R_IR_grad_y)
        joint_int  = torch.maximum(R_vis, R_ir)
        
        
        con_loss = self.l1_loss(Fuse, joint_int)

        gradient_loss = 0.5 * self.l1_loss(Fuse_grad_x, joint_grad_x) + 0.5 * self.l1_loss(Fuse_grad_y, joint_grad_y)

        color_loss = self.l1_loss(Cb_Fuse, Cb_R_vis) + self.l1_loss(Cr_Fuse, Cr_R_vis)

        fusion_loss_total = 0.5 * con_loss  + 1 * gradient_loss  + 1 * color_loss

        return fusion_loss_total

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_index=None):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, preds, targets):
        """
        preds: [N, C, H, W] —— 模型输出 (logits)
        targets: [N, H, W] —— 语义分割标签
        """
        num_classes = preds.shape[1]
        preds = F.softmax(preds, dim=1)

        if self.ignore_index is not None:
            targets = targets.clone()
            targets[targets == self.ignore_index] = 0

        # one-hot
        targets_onehot = F.one_hot(targets, num_classes=num_classes)  # [N, H, W, C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()   # [N, C, H, W]

        if self.ignore_index is not None:
            mask = (targets != self.ignore_index).unsqueeze(1)  # [N, 1, H, W]
            preds = preds * mask
            targets_onehot = targets_onehot * mask

        # dice
        intersection = torch.sum(preds * targets_onehot, dim=(0, 2, 3))
        union = torch.sum(preds + targets_onehot, dim=(0, 2, 3))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score.mean()

        return dice_loss



class MakeLoss(nn.Module):
    def __init__(self, background):
        super(MakeLoss, self).__init__()

        self.semantic_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=background)
        self.dice_loss = DiceLoss(ignore_index=background)
        self.FusionLoss = FusionLoss()

    def forward(self, inputs, outputs, Mask, label):

        input_vis, input_ir = inputs

        out_semantic, out_semantic_r, Fus_img, Fus_img_r = outputs
        
        fusion_loss_total = self.FusionLoss(inputs, Fus_img, Mask)

        fusion_loss_total_cr = self.FusionLoss(inputs, Fus_img_r, Mask)

        semantic_loss_total = self.semantic_loss(out_semantic,label)

        dice_loss_total = self.dice_loss(out_semantic, label)

        dice_loss_total_r = self.dice_loss(out_semantic_r, label)

        ## loss function Lcr
        semantic_loss_total_cr = self.semantic_loss(out_semantic_r, label)

        loss = 1 * semantic_loss_total + 0.5 * fusion_loss_total + 0.25 * semantic_loss_total_cr +  0.25 * fusion_loss_total_cr + dice_loss_total*2 + dice_loss_total_r*0.5

        return loss, semantic_loss_total, fusion_loss_total


class CELoss(nn.Module):
    def __init__(self, background):
        super(CELoss, self).__init__()

        self.semantic_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=background)

    def forward(self, inputs, label):
        semantic_loss_total = self.semantic_loss(inputs,label)

        loss = semantic_loss_total

        return loss
