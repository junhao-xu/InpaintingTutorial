from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import random
import inspect, re
import numpy as np
import os
import collections
import math
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.transform import resize


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def wrapper_gmask(opt):
    # batchsize should be 1 for mask_global
    mask_global = torch.ByteTensor(1, 1, \
                                        opt.fineSize, opt.fineSize)

    res = 0.02  # the lower it is, the more continuous the output will be. 0.01 is too small and 0.1 is too large
    density = 0.25
    MAX_SIZE = 350
    maxPartition = 30
    low_pattern = torch.rand(1, 1, int(res * MAX_SIZE), int(res * MAX_SIZE)).mul(255)
    pattern = F.interpolate(low_pattern, (MAX_SIZE, MAX_SIZE), mode='bilinear').detach()
    low_pattern = None
    pattern.div_(255)
    pattern = torch.lt(pattern, density).byte()  # 25% 1s and 75% 0s
    pattern = torch.squeeze(pattern).byte()
    

    gMask_opts = {}
    gMask_opts['pattern'] = pattern
    gMask_opts['MAX_SIZE'] = MAX_SIZE
    gMask_opts['fineSize'] = opt.fineSize
    gMask_opts['maxPartition'] = maxPartition
    gMask_opts['mask_global'] = mask_global
    return create_gMask(gMask_opts)  # create an initial random mask.


def create_gMask(gMask_opts, limit_cnt=1):
    pattern = gMask_opts['pattern']
    mask_global = gMask_opts['mask_global']
    MAX_SIZE = gMask_opts['MAX_SIZE']
    fineSize = gMask_opts['fineSize']
    maxPartition=gMask_opts['maxPartition']
    if pattern is None:
        raise ValueError
    wastedIter = 0
    while wastedIter <= limit_cnt:
        x = random.randint(1, MAX_SIZE-fineSize)
        y = random.randint(1, MAX_SIZE-fineSize)
        mask = pattern[y:y+fineSize, x:x+fineSize]
        area = mask.sum()*100./(fineSize*fineSize)
        if area>20 and area<maxPartition:
            break
        wastedIter += 1
    if mask_global.dim() == 3:
        mask_global = mask.expand(1, mask.size(0), mask.size(1))
    else:
        mask_global = mask.expand(1, 1, mask.size(0), mask.size(1))

    return mask_global


# inMask is tensor should be bz*1*256*256 float
# Return: ByteTensor
def cal_feat_mask(inMask, nlayers):
    assert inMask.dim() == 4, "mask must be 4 dimensions"
    inMask = inMask.float()
    ntimes = 2**nlayers
    inMask = F.interpolate(inMask, (inMask.size(2)//ntimes, inMask.size(3)//ntimes), mode='nearest')
    inMask = inMask.detach().byte()

    return inMask


# It is only for patch_size=1 for now, although it also works correctly for patchsize > 1.
# For patch_size > 1, we adopt another implementation in `patch_soft_shift/innerPatchSoftShiftTripleModule.py` to get masked region.
# return: flag indicating where the mask is using 1s.
#         flag size: bz*(h*w)
def cal_flag_given_mask_thred(mask, patch_size, stride, mask_thred):
    assert mask.dim() == 4, "mask must be 4 dimensions"
    assert mask.size(1) == 1, "the size of the dim=1 must be 1"
    mask = mask.float()
    b = mask.size(0)
    mask = F.pad(mask, (patch_size//2, patch_size//2, patch_size//2, patch_size//2), 'constant', 0)
    m = mask.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    m = m.contiguous().view(b, 1, -1, patch_size, patch_size)
    m = torch.mean(torch.mean(m, dim=3, keepdim=True), dim=4, keepdim=True)
    mm = m.ge(mask_thred/(1.*patch_size**2)).long()
    flag = mm.view(b, -1)

    return flag


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


"""
   flow: N*h*w*2
        Indicating which pixel will shift to the location.
   mask: N*(h*w)
"""
def highlight_flow(flow, mask):
    """Convert flow into middlebury color code image.
    """
    assert flow.dim() == 4 and mask.dim() == 2
    assert flow.size(0) == mask.size(0)
    assert flow.size(3) == 2
    bz, h, w, _ = flow.shape
    out = torch.zeros(bz, 3, h, w).type_as(flow)
    for idx in range(bz):
        mask_index = (mask[idx] == 1).nonzero()
        img = torch.ones(3, h, w).type_as(flow) * 144.
        u = flow[idx, :, :, 0]
        v = flow[idx, :, :, 1]
        # It is quite slow here.
        for h_i in range(h):
            for w_j in range(w):
                p = h_i*w + w_j
                #If it is a masked pixel, we get which pixel that will replace it.
                # DO NOT USE `if p in mask_index:`, it is slow.
                if torch.sum(mask_index == p).item() != 0:
                    ui = u[h_i,w_j]
                    vi = v[h_i,w_j]
                    img[:, int(ui), int(vi)] = 255.
                    img[:, h_i, w_j] = 200. # Also indicating where the mask is.
        out[idx] = img
    return out


# ################# Style loss #########################
class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        vgg16 = models.vgg16(pretrained=True)

        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # print(self.enc_1)
        # print(self.enc_2)
        # print(self.enc_3)

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):

        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


def gram_matrix(feat):
    (batch, ch, h, w) = feat.size()
    feat = feat.view(batch, ch, h*w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram