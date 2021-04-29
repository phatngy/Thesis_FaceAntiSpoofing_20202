import argparse
import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from model.backbone.repvgg import get_RepVGG_func_by_name, repvgg_model_convert, whole_model_convert
from model_fusion.FaceBagNet_model_FTB_SEFusion import FusionNet

parser = argparse.ArgumentParser(description='RepVGG Conversion')
parser.add_argument('load', metavar='LOAD', help='path to the weights file')
parser.add_argument('save', metavar='SAVE', help='path to the weights file')
parser.add_argument('-a', '--arch', metavar='ARCH', default='RepVGG-A0')


def model_fnc(deploy):
    return FusionNet(deploy=deploy)

def convert():
    args = parser.parse_args()

    repvgg_build_func = get_RepVGG_func_by_name(args.arch)

    train_model = FusionNet(2)

    deploy_model = FusionNet(deploy=True)
    pytorch_total_params = sum(p.numel() for p in deploy_model.parameters())
    print('total_params: ',pytorch_total_params)
    print('Starting converting....')
    if os.path.isfile(args.load):
        print("=> loading checkpoint '{}'".format(args.load))
        checkpoint = torch.load(args.load)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        # ckpt = {k.replace('module.', ''): v for k, v in checkpoint.items()}  # strip the names
        # for (k, v), (k_, _) in zip(checkpoint.items(), train_model.state_dict().items()):
            # print(k,'\t', k_)
        train_model.load_state_dict(checkpoint)
        # exit(0)
    else:
        print("=> no checkpoint found at '{}'".format(args.load))

    whole_model_convert(train_model, deploy_model=deploy_model, save_path=args.save)
    # repvgg_model_convert(train_model, build_func=model_fnc, save_path=args.save)
    return deploy_model
if __name__ == '__main__':
    convert()