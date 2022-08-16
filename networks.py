#-*-coding:utf-8-*-
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torchvision import models
import torch
import functools


from modules.modules import SwitchNorm2d
from modules.shift_unet import UnetGeneratorShiftTriple
from modules.discrimators import NLayerDiscriminator

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = functools.partial(SwitchNorm2d)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    def lambda_rule(epoch):
        lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
        return lr_l
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.normal_(m.weight.data, 0.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


# Note: Adding SN to G tends to give inferior results. Need more checking.
def define_G(input_nc, output_nc, ngf, which_model_netG, opt, mask_global, norm='batch', use_spectral_norm=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    innerCos_list = []
    shift_list = []

    print('input_nc {}'.format(input_nc))
    print('output_nc {}'.format(output_nc))
    print('which_model_netG {}'.format(which_model_netG))

    # Here we need to initlize an artificial mask_global to construct the init model.
    # When training, we need to set mask for special layers(mostly for Shift layers) first.
    # If mask is fixed during training, we only need to set mask for these layers once,
    # else we need to set the masks each iteration, generating new random masks and mask the input
    # as well as setting masks for these special layers.
    print('[CREATED] MODEL')
    # if which_model_netG == 'unet_shift_triple':
    netG = UnetGeneratorShiftTriple(input_nc, output_nc, 8, opt, innerCos_list, shift_list, mask_global, \
                                                         ngf, norm_layer=norm_layer, use_spectral_norm=use_spectral_norm)

    print('[CREATED] MODEL')
    print('Constraint in netG:')
    print(innerCos_list)

    print('Shift in netG:')
    print(shift_list)

    print('NetG:')
    print(netG)

    return init_net(netG, init_type, init_gain, gpu_ids), innerCos_list, shift_list


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, use_spectral_norm=False, init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    # if which_model_netD == 'basic':
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, use_spectral_norm=use_spectral_norm)

    print('NetD:')
    print(netD)
    return init_net(netD, init_type, init_gain, gpu_ids)

