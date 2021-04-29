
import numpy
import torch
from torch import nn

from .backbone.FeatherNet import FeatherNet
BatchNorm2d = nn.BatchNorm2d


class Net(nn.Module):
    def __init__(self, num_class=2, id_class=300, is_first_bn=False):
        super(Net, self).__init__()

        self.is_first_bn = is_first_bn
        if self.is_first_bn:
            self.first_bn = nn.BatchNorm2d(3)
        
        self.encoder = FeatherNet(se=True, avgdown=True)
        encoder_tmp = FeatherNet(se=True, avgdown=True)
        self.ft = encoder_tmp.features

    
    def forward(self, x):
        batch_size,C,H,W = x.shape

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
        x = self.encoder(x)

        return x

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
        x = self.ft(x)
        # x = self.encoder.features(x)
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

# if __name__ == '__main__':
#     import os
#     model = Net(num_class=2, is_first_bn=True)
#     pytorch_total_params = sum(p.numel() for p in model.parameters())
#     print(pytorch_total_params)
#     # print(model)
#     x = torch.randn(1, 3, 48, 48)
#     o = model.forward_res3(x)
#     print(o.size())