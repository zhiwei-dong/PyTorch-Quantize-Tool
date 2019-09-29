# Program:
# This script is for TG_team to quantize their custom neural network pre-trained model.
# It's a general script to use.
# History:
# 2019/09/27    Albert Dong	First release
# License:
# BSD
##########################################################################

# -------------------------    import section(edit if you need)    -------------------------
import argparse

from tqdm import tqdm
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# -------------------------    param section    -------------------------
"""
Args:
    bit_width 位宽
    fraction_length 小数位长度 仅在输入输出量化时起作用
    is_quantize 是否量化 仅在输入输出量化时起作用
    
"""
parser = argparse.ArgumentParser(description='PyTorch Ristretto Quantization Tool')
parser.add_argument('-p', '--pretrain', help='path of pre-trained model')
parser.add_argument('-s', '--saving', help='path to saving quantized model')
parser.add_argument('-b', '--bit_width', type=int, default=8,
                    help='number of bit you want to quantize pretrained model (default:8)')
args = parser.parse_args()
pretrain_model_path = args.pretrain  # 预训练模型参数路径
param_saving_path = args.saving  # 量化后参数存储的位置
bit_width = args.bit_width  # 量化的目标比特数
state = torch.load(pretrain_model_path)  # 预训练模型参数
params = []
fraction_length = numpy.zeros(len(params))
is_quantization = numpy.zeros(len(params))
state_tmp = state.copy()  # copy state for test after quantization
result_param = {}


# -------------------------    override quantization layer component    -------------------------

class EltwiseAdd(nn.Module):
    # __init__ for init params
    # quanti is quantization's abbreviation
    def __init__(self, bit_width, fl, is_quanti, inplace=False):
        super(EltwiseAdd, self).__init__()
        self.inplace = inplace
        self.bit_width = bit_width
        self.fl = fl
        self.is_quanti = is_quanti

    # if is_quanti call trim function else directly return data
    def quantization(self, data):
        if not self.is_quanti:
            return data
        else:
            tmp = Trim2FixedPoint(data, bit_width=self.bit_width, fraction_length=self.fl)
            return tmp

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res += t
        else:
            for t in input[1:]:
                res = self.quantization(res + t)
        return res


class EltwiseMult(nn.Module):
    # __init__ is for init params
    # quanti is quantization's abbreviation
    def __init__(self, bit_width, fl, is_quanti, inplace=False):
        super(EltwiseMult, self).__init__()
        self.inplace = inplace
        self.bit_width = bit_width
        self.fl = fl
        self.is_quanti = is_quanti

    # if is_quanti call trim function else directly return data
    def quantization(self, data):
        if not self.is_quanti:
            return data
        else:
            tmp = Trim2FixedPoint(data, bit_width=self.bit_width, fraction_length=self.fl)
            return tmp.view(data.size())

    def forward(self, *input):
        res = input[0]
        if self.inplace:
            for t in input[1:]:
                res *= t
        else:
            for t in input[1:]:
                res = self.quantization(res * t)
        return res


# -------------------------    model section(custom needed)    -------------------------

class Net(nn.Module):
    def __init__(self, bit_width, fraction_length, is_quantization):
        super(Net, self).__init__()

    def forward(self, x):
        return x


# -------------------------    loader section(custom needed)    -------------------------

class MyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.imgs)


test_data = MyDataset()
data_loader = DataLoader(dataset=test_data, batch_size=args.batch_size)


# -------------------------    eval section(custom needed)    -------------------------

def evaluate(model, data_loader):
    model.eval()
    model.cuda()
    accuracy = 0
    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            pass
    return accuracy


# -------------------------    quantization function section    -------------------------
# data 原数据
# bit_width 位宽
# fraction_length 小数部分长度

def Trim2FixedPoint(data, bit_width=8, fraction_length=0):
    max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fraction_length)  # 当前设定位宽和fl的最大值
    min_data = -pow(2, bit_width - 1) * pow(2, -fraction_length)  # 当前设定位宽和fl的最小值
    # https://pytorch.org/docs/stable/torch.html?highlight=clamp#torch.clamp
    data = torch.clamp(data, min_data, max_data)  # 设置上下极限，使得原数据被限制在本动态范围内
    data /= pow(2, -fraction_length)  # 除分数部分
    data = torch.floor(data)  # 得到小的最接近的整数d
    data[data % 2 != 0] += 1  # ？
    data *= pow(2, -fraction_length)  # 乘回去
    return data


# -------------------------    quantize params    -------------------------
"""
这个部分量化所有的参数，逐层量化，得到最佳量化策略之后测试精度并且保存最佳量化策略下的模型
"""
for layer in tqdm(range(len(params)), ascii=True):  # 遍历所有的层数
    acc_param = 0  # init accuracy
    print('-------Quantizing layer:{}\'s parameter-------'.format(layer))
    for fraction_length_of_param in range(bit_width):  # 遍历所有的小数位置
        print('--------Trying fraction length: {}---------'.format(fraction_length_of_param))
        for key in params[layer]:  # 量化制定的层
            param = state[key].clone()  # 提取特定层参数
            param = Trim2FixedPoint(param.float(), bit_width, fraction_length_of_param)  # 量化
            state_tmp[key] = param  # 修改临时参数中的指定层参数
        model = Net(bit_width=bit_width,
                    fraction_length=fraction_length,
                    is_quantization=is_quantization)  # 模型实例化
        model.load_state_dict(state_tmp)  # 加载参数
        acc_param_eval = evaluate(model, data_loader)  # eval
        if acc_param_eval > acc_param:
            result_param[layer] = [fraction_length_of_param, acc_param_eval]  # 保存小数位置和精度
        else:
            result_param[layer] = result_param[layer]
            # 如果没能获取更高的精度，恢复最好的参数
            for key_new in params[layer]:
                param_new = state[key_new].clone()
                param_new = Trim2FixedPoint(param_new.float(), bit_width, result_param[layer][0])
                state_tmp[key_new] = param_new
        acc_param = max(acc_param, acc_param_eval)
        print('--------Accuracy of fraction length: {} is {}---------'.format(fraction_length_of_param, acc_param))
    print('-------Layer:{} parameter\'s best result is {} -------'.format(layer, result_param[layer][0]))
final_state = state.copy()
# 使用最佳量化策略，量化预训练模型
for layer_num, _ in result_param.items():
    for key in params[layer_num]:
        param = state[key].clone()
        param = Trim2FixedPoint(param.float(), bit_width, result_param[layer_num][0])
        final_state[key] = param
model_param = Net(bit_width=bit_width,
                  fraction_length=fraction_length,
                  is_quantization=is_quantization)  # 实例化模型
model_param.load_state_dict(final_state)  # eval
acc_param_eval = evaluate(model_param, data_loader)  # get eval accuracy
print('-------Quantize parameter is done, best accuracy is {} -------'.format(acc_param_eval))

# -------------------------    quantize input && output    -------------------------
"""
这个部分是为了获得 fraction_length，这个参数是为模型定义量化的时候准备的
"""
for layer in tqdm(range(len(is_quantization)), ascii=True):  # 遍历所有层
    print('-------Quantizing layer:{}\'s inout-------'.format(layer))
    acc_param = 0  # init accuracy
    for fraction_length_of_param in range(bit_width):  # 遍历所有的小数位
        print('--------Trying fraction length: {}---------'.format(fraction_length_of_param))
        fraction_length[layer] = fraction_length_of_param
        is_quantization[layer] = 1
        model = Net(bit_width=bit_width,
                    fraction_length=fraction_length,
                    is_quantization=is_quantization)
        model.load_state_dict(final_state)
        acc_inout_eval = evaluate(model, data_loader)
        fraction_length[layer] = fraction_length_of_param if acc_inout_eval > acc_param else fraction_length[layer]
        acc_param = max(acc_param, acc_inout_eval)
# test section
print('-------Testing-------')
is_quantization = numpy.ones_like(fraction_length)  # 返回一个和best一样尺寸的全1矩阵
model = Net(bit_width=bit_width,
            fraction_length=fraction_length,
            is_quantization=is_quantization)
model.load_state_dict(final_state)
acc_inout_eval = evaluate(model, data_loader)
print('-------Quantize inout is done, best accuracy is {}-------'.format(acc_inout_eval))

# -------------------------    saving quantized model    -------------------------
print("Saving quantized model.")
torch.save(final_state, param_saving_path)  # 保存最佳策略下的参数
print("Quantization finished!")
