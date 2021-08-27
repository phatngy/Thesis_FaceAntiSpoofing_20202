
import numpy
import torch
from torch import nn

from .backbone.repvgg_new import *
BatchNorm2d = nn.BatchNorm2d


class Net(nn.Module):
    def __init__(self, num_class=2, id_class=300, is_first_bn=False, deploy=False):
        super(Net, self).__init__()

        self.is_first_bn = is_first_bn
        if self.is_first_bn:
            self.first_bn = nn.BatchNorm2d(3)
        
        # self.encoder = create_RepVGG_A0(deploy=deploy)
        encoder_tmp = create_RepVGG_A0(deploy=deploy)
        self.stage0 = encoder_tmp.stage0
        self.stage1 = encoder_tmp.stage1
        self.stage2 = encoder_tmp.stage2
        # self.stage3 = encoder_tmp.stage3
        # self.stage4 = encoder_tmp.stage4
        # self.gap = encoder_tmp.gap
        # self.linear = encoder_tmp.linear
    
    # def forward(self, x):
    #     batch_size,C,H,W = x.shape

    #     if self.is_first_bn:
    #         x = self.first_bn(x)
    #     else:
    #         mean=[0.485, 0.456, 0.406] #rgb
    #         std =[0.229, 0.224, 0.225]

    #         x = torch.cat([
    #             (x[:,[0]]-mean[0])/std[0],
    #             (x[:,[1]]-mean[1])/std[1],
    #             (x[:,[2]]-mean[2])/std[2],
    #         ],1)
    #     # x = self.encoder(x)

    #     out = self.stage0(x)
    #     out = self.stage1(out)
    #     out = self.stage2(out)
    #     out = self.stage3(out)
    #     out = self.stage4(out)
    #     out = self.gap(out)
    #     out = out.view(out.size(0), -1)
    #     out = self.linear(out)
    #     return x

    def forward_res3(self, x):
        batch_size, C, H, W = x.shape

        if self.is_first_bn:
            x = self.first_bn(x)
        else:
            mean=[0.485, 0.456, 0.406] #rgb
            std =[0.229, 0.224, 0.225]

            x = torch.cat([
                (x[:,[0]]-mean[0])/std[0],
                (x[:,[1]]-mean[1])/std[1],
                (x[:,[2]]-mean[2])/std[2],
            ],1)
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return x        
    
    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['backup']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, BatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False
