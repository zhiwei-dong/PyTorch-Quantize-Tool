# coding=utf-8
# import argparse
#
# # test parser
# parser = argparse.ArgumentParser(description='PyTorch Ristretto Quantization Tool')
# # bit_width 位宽
# # fraction_length 小数位长度 仅在输入输出量化时起作用
# # is_quantize 是否量化 仅在输入输出量化时起作用
# parser.add_argument('-p', '--pretrain', help='location of pre-trained model')
# parser.add_argument('-s', '--saving', help='location to saving quantized model')
# parser.add_argument('-b', '--bit_width', type=int, default=8,
#                     help='number of bit you want to quantize pretrained model (default:8)')
# args = parser.parse_args()
#
# pretrain_model_path = args.pretrain  # pre-trained param
# param_saving_path = args.saving  # 量化后参数存储的位置
# bit_width = args.bit_width
#
# print(pretrain_model_path)
# print(param_saving_path)
# print(bit_width)

# test param

import torchvision.models as models
import torch

path = "/Users/dongz/Dongz/dl/resnet50-19c8e357.pth"
resnet50 = models.resnet50()
pretrain_model = torch.load(path)


def get_params(state):
    params = []
    tmplist = []
    tmpname = ''
    for name in state:
        if tmpname == '':
            tmpname = name.split('.')[0:-1]
            tmplist.append(name)
        elif tmpname == name.split('.')[0:-1]:
            tmplist.append(name)
        else:
            params.append(tmplist)
            tmplist = []
            tmpname = name.split('.')[0:-1]
            tmplist.append(name)
    return params


params = get_params(pretrain_model)

print(pretrain_model)

# test for quantization component
