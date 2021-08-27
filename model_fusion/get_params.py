
import sys
sys.path.append("..")
from FaceBagNet_model_FTB_SEFusion import FusionNet
# from FaceBagNet_model_A_SEFusion import FusionNet
# from FacebagNet_model_RepVGG_SEFusion import FusionNet

from model.backbone.repvgg_new import RepVGGBlock, repvgg_model_convert
from torchsummary import summary
import torch
from prettytable import PrettyTable
from pytorch_modelsize import SizeEstimator

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(f"Total Trainable Params: {total_params}")
    print(table)
    return total_params

model = FusionNet(2)
dummy = torch.rand(36, 9, 48, 48)
# output = model.forward(dummy)
converted_model = repvgg_model_convert(model)
p = count_parameters(model)
print(p)
se = SizeEstimator(model, input_size=(36, 9, 48, 48))
print(se.estimate_size())
# p1 = count_parameters(converted_model)
# print(p1)
# # converted_model.train()
print(summary(model.cuda(), (9, 48, 48)))
# print(summary(converted_model.cuda(), (9, 48, 48)))
