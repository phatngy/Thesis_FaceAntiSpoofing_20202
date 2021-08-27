import torch
import sys
sys.path.append("..")
from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from model.FaceBagNet_model_FTB import Net
from model.backbone.FeatherNet import SELayer
from model.backbone.FaceBagNet import SEModule
from model.backbone.repvgg_new import RepVGGBlock
from model.backbone.FaceBagNet import SEModule, SEResNeXtBottleneck

BatchNorm2d = nn.BatchNorm2d


class FusionNet(nn.Module):
    def __init__(self, num_class=2, deploy=False, width_multiplier=[1.5, 1.5, 1.5, 2.75], num_blocks=[2, 4, 2, 2], override_groups_map=None):
        super(FusionNet, self).__init__()
        self.deploy = deploy
        width_multiplier = [1, 1, 1, 1]
        self.cur_layer_idx = 1
        self.in_planes = int(256 * width_multiplier[2]) 
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.color_module = Net(num_class=num_class, is_first_bn=True)
        self.depth_module = Net(num_class=num_class, is_first_bn=True)
        self.ir_module = Net(num_class=num_class, is_first_bn=True)

        self.color_SE = SELayer(64,reduction=8)
        self.depth_SE = SELayer(64,reduction=8)
        self.ir_SE = SELayer(64,reduction=8)

        self.bottleneck = nn.Sequential(nn.Conv2d(64*3, int(256 * width_multiplier[2]), kernel_size=1, padding=0),
                                         nn.BatchNorm2d(int(256 * width_multiplier[2])),
                                         nn.ReLU(inplace=True))

        # self.res_0 = self._make_layer(
        #     SEResNeXtBottleneck,
        #     planes=256,
        #     blocks=2,
        #     stride=2,
        #     groups=32,
        #     reduction=16,
        #     downsample_kernel_size=1,
        #     downsample_padding=0
        # )
        # self.res_1 = self._make_layer(
        #     SEResNeXtBottleneck,
        #     planes=512,
        #     blocks=2,
        #     stride=2,
        #     groups=32,
        #     reduction=16,
        #     downsample_kernel_size=1,
        #     downsample_padding=0
        # )
        self.res_0 = self._make_RepVGG_layer(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.res_1 = self._make_RepVGG_layer(int(512 * width_multiplier[3]), num_blocks[3], stride=2)

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(int(512 * width_multiplier[3]), 256),
                                # nn.Linear(1024, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, num_class))
    
    def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
                    downsample_kernel_size=1, downsample_padding=0):
        downsample = None
        self.inplanes = planes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=downsample_kernel_size, stride=stride,
                          padding=downsample_padding, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, groups, reduction, stride,
                            downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups, reduction))

        return nn.Sequential(*layers)
    
    def _make_RepVGG_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            cur_groups = self.override_groups_map.get(self.cur_layer_idx, 1)
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=cur_groups, deploy=self.deploy))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)

    def forward(self, x):
        batch_size,C,H,W = x.shape

        color = x[:, 0:3,:,:]
        depth = x[:, 3:6,:,:]
        ir = x[:, 6:9,:,:]

        color_feas = self.color_module.forward_res3(color)
        depth_feas = self.depth_module.forward_res3(depth)
        ir_feas = self.ir_module.forward_res3(ir)

        color_feas = self.color_SE(color_feas)
        depth_feas = self.depth_SE(depth_feas)
        ir_feas = self.ir_SE(ir_feas)

        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        fea = self.bottleneck(fea)
        x = self.res_0(fea)
        x = self.res_1(x)
        ft = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        x = self.fc(ft)
        return x, ft, self.fc[:3](ft)

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


if __name__ == "__main__":
    model = FusionNet(2)
    dummy = torch.rand(36, 9, 48, 48)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total_params: ',pytorch_total_params)
    output = model.forward(dummy)
    # torch.onnx.export(model, dummy, 'FaceBagNet_model_FTB_SEFusion.onnx', verbose=False) 