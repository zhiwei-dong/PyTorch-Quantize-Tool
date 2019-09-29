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

import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

# -------------------------    param section    -------------------------

parser = argparse.ArgumentParser(description='PyTorch Ristretto Quantization Tool')
# bw bit_width 位宽
# fl fraction_length 小数位长度
# bool_q => is_quantize 是否量化
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
model_q_path = './model/model_quan_704_xh_src.pth'  # pre-trained param
model_param_path = './model/model_quan_704_xh_param.pth'  # 量化后的位置
state = torch.load(model_q_path)  # pre-trained state
result_param = {}
params = []
state_tmp = state.copy()  # copy state for test after quantization
bw_param = 8
bw = 8
fl = numpy.zeros(shape=(11))
bool_q = numpy.zeros(shape=(11))

args = parser.parse_args()


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
            tmp = Trim2FixedPoint(data, bit_width=self.bit_width, fl=self.fl)
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
            tmp = Trim2FixedPoint(data, bit_width=self.bit_width, fl=self.fl)
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
# fl fraction_length 小数部分长度

def Trim2FixedPoint(data, bit_width=8, fl=0):
    max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fl)  # 当前设定位宽和fl的最大值
    min_data = -pow(2, bit_width - 1) * pow(2, -fl)  # 当前设定位宽和fl的最小值
    # https://pytorch.org/docs/stable/torch.html?highlight=clamp#torch.clamp
    data = torch.clamp(data, min_data, max_data)  # 设置上下极限，使得原数据被限制在本动态范围内
    data /= pow(2, -fl)  # 除分数部分
    data = torch.floor(data)  # 得到小的最接近的整数d
    data[data % 2 != 0] += 1  # ？
    data *= pow(2, -fl)  # 乘回去
    return data


# -------------------------    quantize params    -------------------------

for layer in range(len(params)):  # 遍历所有的层数
    acc_param = 0  # init acc
    for fli_param in range(bw_param):  # 遍历所有的位宽
        print('-------Quantize layer:{} parameter-------'.format(layer))
        print('--------Quantize bit: {}---------'.format(fli_param))
        for key in params[layer]:  # 量化制定的层
            param = state[key].clone()
            param = Trim2FixedPoint(param.float(), bw_param, fli_param)
            state_tmp[key] = param
        model_param = Net(bit_width=bw, fraction_length=fl, is_quantization=bool_q)
        model_param.load_state_dict(state_tmp)
        acc_param_new = evaluate(model_param, data_loader)  # test

        if acc_param_new > acc_param:
            result_param[layer] = [fli_param, acc_param_new]
        else:
            result_param[layer] = result_param[layer]
            for key_new in params[layer]:
                param_new = state[key_new].clone()
                param_new = Trim2FixedPoint(param_new.float(), bw_param, result_param[layer][0])
                state_tmp[key_new] = param_new

        acc_param = max(acc_param, acc_param_new)
        print('acc_param:', acc_param)
    print('result_param:', result_param)
state_test = state.copy()
bw = 8
for k, v in result_param.items():
    for key in params[k]:
        param = state[key].clone()
        param = Trim2FixedPoint(param.float(), bw, result_param[k][0])
        state_test[key] = param
torch.save(state_test, model_param_path)  # 保存最佳策略下的参数
model_param = Net(bit_width=bw, fraction_length=fl, is_quantization=bool_q)
model_param.load_state_dict(state_test)
acc_param_new = evaluate(model_param, data_loader)
print(acc_param_new)  # 最佳策略下的精度

# -------------------------    quantize input & output    -------------------------

result = numpy.zeros(shape=(11))
for layer in range(len(bool_q)):  # 遍历层数
    new_fl = fl
    new_bool_q = bool_q  # .copy()
    acc = 0
    for fli in range(bw):
        print('-------Quantize layer:{}-------'.format(layer))
        print('-------Quantize bit:{}-------'.format(fli))
        print('-------Test-------')
        new_fl[layer] = fli
        new_bool_q[layer] = 1
        print('new_fl', new_fl)
        print('new_bool_q', new_bool_q)
        model = Net(bit_width=bw, fraction_length=fl, is_quantization=bool_q)
        model.load_state_dict(state)
        acc_new = evaluate(model, data_loader)
        result[layer] = fli if acc_new > acc else result[layer]
        new_fl[layer] = result[layer]
        acc = max(acc, acc_new)
        print(acc_new)
best = result
bool_q_new = numpy.ones_like(best)
bw = 8
model = Net(bit_width=bw, fraction_length=fl, is_quantization=bool_q)
model.load_state_dict(state)
acc = evaluate(model, data_loader)

# -------------------------    saving quantized model    -------------------------

