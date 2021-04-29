import torch

from torch import nn
import torch.nn.functional as F
from torchvision.models.resnet import BasicBlock

from model.FaceBagNet_model_RepVGG import Net
from model.backbone.FaceBagNet import SEModule
from model.backbone.repvgg import RepVGGBlock
BatchNorm2d = nn.BatchNorm2d


class FusionNet(nn.Module):
    def __init__(self, num_class=2, deploy=False, width_multiplier=[0.75, 0.75, 0.75, 2.5], num_blocks=[2, 4, 14, 1], override_groups_map=None):
        super(FusionNet, self).__init__()
        self.deploy = deploy
        self.cur_layer_idx = 1
        self.in_planes = int(256 * width_multiplier[2]) 
        self.override_groups_map = override_groups_map or dict()

        assert 0 not in self.override_groups_map

        self.color_module = Net(num_class=num_class, is_first_bn=True)
        self.depth_module = Net(num_class=num_class, is_first_bn=True)
        self.ir_module = Net(num_class=num_class, is_first_bn=True)

        self.color_SE = SEModule(96,reduction=8)
        self.depth_SE = SEModule(96,reduction=8)
        self.ir_SE = SEModule(96,reduction=8)

        self.bottleneck = nn.Sequential(nn.Conv2d(96*3, int(256 * width_multiplier[2]), kernel_size=1, padding=0),
                                         nn.BatchNorm2d(int(256 * width_multiplier[2])),
                                         nn.ReLU(inplace=True))

        self.res_0 = self._make_RepVGG_layer(int(256 * width_multiplier[2]), num_blocks[2], stride=2)
        self.res_1 = self._make_RepVGG_layer(int(512 * width_multiplier[3]), num_blocks[3], stride=2)

        self.fc = nn.Sequential(nn.Dropout(0.5),
                                nn.Linear(int(512 * width_multiplier[3]), 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, num_class))
    
    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 :
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        # print(color_feas.size())
        # exit(0)
        color_feas = self.color_SE(color_feas)
        depth_feas = self.depth_SE(depth_feas)
        ir_feas = self.ir_SE(ir_feas)

        fea = torch.cat([color_feas, depth_feas, ir_feas], dim=1)
        fea = self.bottleneck(fea)
        x = self.res_0(fea)
        x = self.res_1(x)
        x = F.adaptive_avg_pool2d(x, output_size=1).view(batch_size, -1)
        x = self.fc(x)
        return x, None, None

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
    output = model.forward(dummy)
    # torch.onnx.export(model, dummy, 'FaceBagNet_model_FTB_SEFusion.onnx', verbose=False) 