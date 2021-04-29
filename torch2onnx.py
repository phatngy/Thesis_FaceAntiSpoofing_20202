import torch
from model_fusion.FaceBagNet_model_FTB_SEFusion import FusionNet


model = FusionNet(2)
model.eval()
dummy = torch.randn(36, 9, 32, 32)
o = model.forward(dummy)


print(o.size())
torch.onnx.export(model, dummy, 'FaceBagNet_model_FTB_SEFusion.onnx', verbose=False)
