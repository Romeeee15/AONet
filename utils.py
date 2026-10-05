import os
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def check_dir_exist(dir):
    """create directories"""
    if os.path.exists(dir):
        return
    else:
        names = os.path.split(dir)
        dir = ''
        for name in names:
            dir = os.path.join(dir,name)
            if not os.path.exists(dir):
                try:
                    os.mkdir(dir)
                except:
                    pass
        print('dir','\''+dir+'\'','is created.')

# def cal_Dice(img1,img2):
#     shape = img1.shape
#     I = 0
#     U = 0
#     for i in range(shape[0]):
#         for j in range(shape[1]):
#             if img1[i,j] >= 1 and img2[i,j] >= 1:
#                 I += 1
#             if img1[i,j] >= 1 or img2[i,j] >= 1:
#                 U += 1
#     return 2*I/(I+U+1e-5)


# def cal_miou(img1,img2):
#     classnum = img2.max()
#     iou=np.zeros((int(classnum),1))
#     for i in range(int(classnum)):
#         imga=img1==i+1
#         imgb=img2==i+1
#         imgi=imga * imgb
#         imgu=imga + imgb
#         iou[i]=np.sum(imgi)/np.sum(imgu)
#     miou=np.mean(iou)
#     return miou


def cal_Dice(pred, gt, num_classes=5):
    """
    计算一个 batch 的 mDSC
    :param pred: 预测结果，形状为 [batch, h, w]，NumPy 数组
    :param gt: 真实标签，形状为 [batch, h, w]，NumPy 数组
    :param num_classes: 类别数量
    :return: mDSC
    """
    dsc_list = []
    for class_idx in range(num_classes):
        # 将预测和真实标签转换为当前类别的二值表示
        pred_class = (pred == class_idx).astype(int)
        gt_class = (gt == class_idx).astype(int)  # 修改为 NumPy 的 astype(int)

        # 计算交集和两者的总和
        intersection = np.sum(pred_class * gt_class, axis=(1, 2))  # 按 batch 计算交集
        total = np.sum(pred_class, axis=(1, 2)) + np.sum(gt_class, axis=(1, 2))

        # 计算 DSC
        dsc = (2 * intersection / (total + 1e-6)).mean()  # 防止除零
        dsc_list.append(dsc)

    # 计算每一类的平均 DSC
    mDSC = np.mean(dsc_list)
    return mDSC


def cal_miou(pred, gt, num_classes=5):
    """
    计算一个 batch 的 mIoU
    :param pred: 预测结果，形状为 [batch, h, w]，NumPy 数组
    :param gt: 真实标签，形状为 [batch, h, w]，NumPy 数组
    :param num_classes: 类别数量
    :return: mIoU
    """
    iou_list = []
    for class_idx in range(num_classes):
        # 将预测和真实标签转换为当前类别的二值表示
        pred_class = (pred == class_idx).astype(int)
        gt_class = (gt == class_idx).astype(int)  # 修改为 NumPy 的 astype(int)

        # 计算交集和并集
        intersection = np.sum(pred_class * gt_class, axis=(1, 2))  # 按 batch 计算交集
        union = np.sum(pred_class, axis=(1, 2)) + np.sum(gt_class, axis=(1, 2)) - intersection

        # 计算 IoU
        iou = (intersection / (union + 1e-6)).mean()  # 防止除零
        iou_list.append(iou)

    # 计算每一类的平均 IoU
    mIoU = np.mean(iou_list)
    return mIoU


def cal_acc(img1,img2):
    shape = img1.shape
    acc = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            if img1[i,j] == img2[i,j]:
                acc += 1
    return acc/(shape[0]*shape[1])


def make_one_hot(input, shape):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    result = torch.zeros(shape)
    result.scatter_(1, input.cpu(), 1)
    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1).cpu()
        target = target.contiguous().view(target.shape[0], -1).cpu()

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        shape=predict.shape
        target = torch.unsqueeze(target, 1)
        target=make_one_hot(target.long(),shape)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss / target.shape[1]

def get_patch_random(data,label, cube_size, patch_size):
    patch_pos = []
    for i in range(3):
        patch_pos.append(torch.randint(0, cube_size[i] - patch_size[i] + 1, (1,)))
        #print(cube_size[i], patch_size[i], patch_pos[i])
    data_crop = data[:, :, patch_pos[0]:patch_pos[0] + patch_size[0], patch_pos[1]:patch_pos[1] + patch_size[1],patch_pos[2]:patch_pos[2] + patch_size[2]]
    data_crop = data_crop.contiguous()
    label_crop = label[:, :, patch_pos[1]:patch_pos[1] + patch_size[1],patch_pos[2]:patch_pos[2] + patch_size[2]]
    label_crop  = label_crop.contiguous()
    return data_crop,label_crop

def split_test(data,model, cube_size, patch_size,n_classes):
    outshape=[1,n_classes,1,cube_size[1],cube_size[2]]
    result = torch.zeros(outshape)
    result = result.to(data.device)
    for x in range(0, cube_size[0], patch_size[0]):
        for y in range(0, cube_size[1], patch_size[1]):
            for z in range(0, cube_size[2], patch_size[2]):
                input=data[:,:,x:x+patch_size[0],y:y+patch_size[1],z:z+patch_size[2]]
                output=model(input)
                result[:,:,x:x+patch_size[0],y:y+patch_size[1],z:z+patch_size[2]]=output
    return result