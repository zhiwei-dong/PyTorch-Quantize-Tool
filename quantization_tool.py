# coding=utf-8
# Program:
# This script is for TG_team to quantize their custom neural network pre-trained model.
# It's a general script to use.
# History:
# 2019/09/27    Albert Dong	@ First release
# 2019/10/9     Albert Dong	@ add get model params
# 2019/10/13    Albert Dong @ add muti_gpu support
# 2019/10/13    Albert Dong @ remove tqdm support
# 2019/10/13    Albert Dong @ re-add tqdm support
# 2019/10/13    Albert Dong @ print params
#
# License:
# BSD
##########################################################################

# -------    import section(edit if you need)    -------
import argparse
import os

import cv2
import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

import utils


# -------    param section    -------
def bn2scale(model_state, layer_name):
    """
    description:
        将指定bn层的函数改成scale层的形式，解决由PyTorch BN层的四个参数数量级不一致导致的在量化时的精度
        较大损失，返回值为包含处理好的bn层的模型参数。
        PyTorch BN数学公式：out = gamma * (x - mean)/sqrt(var) + beta
        Caffe Scale 数学公式：out = alpha * x + beta
        通过使得 new_gamma = gamma / sqrt(var),
               new_beta = beta - gamma * mean / sqrt(var)
               new_mean = 0, new_var = 1 来完成变换。P.S. 为了利用原来的BN的结构来实现scale。
    :parameter
        model_state:
            未处理过的模型参数
        layer_name: str
            bn层的参数存储名字的layername，如cnn.batchnorm0，如果全名为cnn.batchnorm0.weight
    :return:
        model_state:
            处理过的模型参数
    """
    # 从未处理的模型中获取bn层的四个参数
    weight = model_state.get(layer_name + ".weight")
    bias = model_state.get(layer_name + ".bias")
    mean = model_state.get(layer_name + ".running_mean")
    var = model_state.get(layer_name + ".running_var")
    # 计算标准差，即sqrt(var) => std
    std = torch.sqrt(var)
    # 计算新的值
    a = weight / std
    b = bias - weight * mean / std
    e = torch.zeros(mean.shape)
    s = torch.ones(var.shape)
    # 保存参数到模型
    model_state[layer_name + ".weight"] = a
    model_state[layer_name + ".bias"] = b
    model_state[layer_name + ".running_mean"] = e
    model_state[layer_name + ".running_var"] = s


def get_params(model_state, is_scale):
    """
    description:
        这个函数是暂时是专用函数，只能针对名称为layer_name.params_name 这种命名方式的提取。
        比如："cnn.conv0.weight" 中 cnn.conv0 是layer_name, weight 是 param_name
    :param model_state: 预训练模型参数
    :param is_scale: 是否修改所有bn层到scale层表示
    :return:
        params_list: 可量量化的参数列表
    """
    # 初始化参数
    params_list = []
    for _, key in enumerate(model_state):
        # 获取名称
        layer_name = key.split('.')[0:-1]
        param_name = key.split('.')[-1]
        # 这里的判定还需要优化
        # 如果指定需要优化，且当前层为bn层，且当前遍历参数为权重（只计算一次，防止出错）
        # 调用bn2scale函数处理bn层。
        if is_scale and 'batchnorm' in ''.join(layer_name) and 'weight' in ''.join(param_name):
            bn2scale(model_state, '.'.join(layer_name))
        # 如果 _ (index) 为0，说明列表中还什么都没有，将key.split()追加到列表后面
        # 这是因为 key是一个字符串，key.split() 则是一个list
        # 比如key = 'hello' key.split() = ['hello']
        # 而 list(key) = ['h', 'e', 'l', 'l', 'o']
        # 或者当前的layer_name 并非在现在的列表的最后一项list中，追加key.split()
        if not _ or layer_name != params_list[-1][-1].split('.')[0:-1]:
            params_list.append(key.split())
        # 在其他的情况下，如果param_name不是"num_batches_tracked" 则追加到列表的最后一项中
        elif param_name != "num_batches_tracked":
            params_list[-1].append(key)
    return params_list


"""
Args:
    bit_width 位宽
    fraction_length 小数位长度 仅在输入输出量化时起作用
    is_quantize 是否量化 仅在输入输出量化时起作用
    
"""
parser = argparse.ArgumentParser(description='PyTorch Ristretto Quantization Tool')
parser.add_argument('-p', '--pretrain', help='path of pre-trained model',
                    default=
                    '/home/yzzc/Work/lq/license_plate_pytorch/crnn_chinese_characters_rec/expr/all_ft_2/crnn_best.pth')
parser.add_argument('-s', '--saving', help='path to saving quantized model',
                    default=
                    '/home/yzzc/Work/lq/license_plate_pytorch/crnn_chinese_characters_rec/expr/all_ft_2/crnn_best_quantize.pth')
parser.add_argument('-b', '--bit_width', type=int, default=8,
                    help='number of bit you want to quantize pretrained model (default:8)')
parser.add_argument('--gpu_id', default='0,1,2,3', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--bn2scale', default=True, type=bool,
                    help='Transfer BN to Scale')
args = parser.parse_args()

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
assert torch.cuda.is_available(), 'CUDA is needed for CNN'

pretrain_model_path = args.pretrain  # 预训练模型参数路径
param_saving_path = args.saving  # 量化后参数存储的位置
bit_width = args.bit_width  # 量化的目标比特数
state = torch.load(pretrain_model_path)  # 预训练模型参数
params = get_params(state, args.bn2scale)
fraction_length = numpy.zeros(len(params))
is_quantization = numpy.zeros(len(params))
state_tmp = state.copy()  # copy state for test after quantization
result_param = params.copy()  # copy params structure for record

# -------    echo program params    -------

print("gpu id: " + str(args.gpu_id))
print("bit width: " + str(args.bit_width))
print("saving path: " + str(args.saving))
print("pretrain path: " + str(args.pretrain))
print("transfer BN to Scale: " + str(args.bn2scale))

# -------    warning message    -------

print("Please Notify: You MUST be sure that you have permission of saving path! ")


# -------    override quantization layer component    -------

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


# -------    model section(custom needed)    -------

class Quantization(nn.Module):
    def __init__(self, bit_width, fraction_length, bool_q):
        super(Quantization, self).__init__()
        self.bit_width = bit_width
        self.fraction_length = fraction_length
        self.bool_q = bool_q

    def forward(self, data):

        if not self.bool_q:
            return data
        else:
            new_data = Trim2FixedPoint(data, bit_width=self.bit_width, fraction_length=self.fraction_length)
            return new_data


'''
# Example for model defination
class Net(nn.Module):
    def __init__(self, bit_width, fraction_length, is_quantization):
        super(Net, self).__init__()

    def forward(self, x):
        return x
'''


class Net(nn.Module):
    '''
    #original model
    def __init__(self, bit_width, fraction_length, is_quantization, imgH=32, nc=1, nclass=76, leakyRelu=True):
        super(Net, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3, 2, 2]
        ps = [1, 1, 1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1, 2]
        nm = [64, 128, 256, 256, 512, 512, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))

            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3, True)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 0)))
        convRelu(4, True)
        convRelu(5, True)

        branch1 = nn.Sequential()
        branch2 = nn.Sequential()

        branch1.add_module('pooling{0}'.format(3),nn.MaxPool2d((2, 2), (2, 1), (0, 0)))
        # branch1.add_module('quanti_1',Quantization(bit_width, fraction_length[0], bool_q[0]))
        branch1.add_module('conv7',nn.Conv2d(nm[5], nm[6], ks[6], ss[6], ps[6]))
        branch1.add_module('batchnorm7', nn.BatchNorm2d(nm[6]))
        branch1.add_module('relu7', nn.ReLU(True))
        branch1.add_module('conv8',nn.Conv2d(nm[6], nm[7], ks[7], ss[7], ps[7]))
        branch1.add_module('batchnorm8', nn.BatchNorm2d(nm[7]))
        branch1.add_module('relu8', nn.ReLU(True))
        branch2.add_module('pooling7',nn.MaxPool2d(2, 2))
        branch2.add_module('conv9',nn.Conv2d(nm[7], nm[8], ks[8], ss[8], ps[8]))
        branch2.add_module('batchnorm9', nn.BatchNorm2d(nm[8]))
        branch2.add_module('relu9', nn.ReLU(True))

        self.cnn = cnn
        self.branch1 = branch2 #first two character branch
        self.branch2 = branch1 #last five-six character branch
        self.fc1 = nn.Linear(512*1, 76)
        self.fc2 = nn.Linear(512*1, 76)
    '''

    def __init__(self, bit_width, fraction_length, is_quantization, imgH=32, nc=1, nclass=76, leakyRelu=True):
        super(Net, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 3, 2, 2]
        ps = [1, 1, 1, 1, 1, 1, 1, 0, 0]
        ss = [1, 1, 1, 1, 1, 1, 1, 1, 2]
        nm = [64, 128, 256, 256, 512, 512, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            cnn.add_module('quanti{0}'.format(2 * i),
                           Quantization(bit_width, fraction_length[2 * i], is_quantization[2 * i]))

            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn.add_module('quanti{0}'.format(2 * i + 1),
                               Quantization(bit_width, fraction_length[2 * i], is_quantization[2 * i]))

            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))

            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0, True)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))
        convRelu(1, True)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))
        convRelu(2, True)
        convRelu(3, True)
        cnn.add_module('pooling{0}'.format(2), nn.MaxPool2d((2, 2), (2, 1), (0, 0)))
        convRelu(4, True)
        convRelu(5, True)

        branch1 = nn.Sequential()
        branch2 = nn.Sequential()

        branch1.add_module('pooling{0}'.format(3), nn.MaxPool2d((2, 2), (2, 1), (0, 0)))
        # branch1.add_module('quanti_1',Quantization(bit_width, fraction_length[0], bool_q[0]))
        branch1.add_module('conv7', nn.Conv2d(nm[5], nm[6], ks[6], ss[6], ps[6]))
        branch1.add_module('quanti{0}'.format(12), Quantization(bit_width, fraction_length[12], is_quantization[12]))

        branch1.add_module('batchnorm7', nn.BatchNorm2d(nm[6]))
        branch1.add_module('quanti{0}'.format(13), Quantization(bit_width, fraction_length[13], is_quantization[13]))

        branch1.add_module('relu7', nn.ReLU(True))
        branch1.add_module('conv8', nn.Conv2d(nm[6], nm[7], ks[7], ss[7], ps[7]))
        branch1.add_module('quanti{0}'.format(14), Quantization(bit_width, fraction_length[14], is_quantization[14]))

        branch1.add_module('batchnorm8', nn.BatchNorm2d(nm[7]))
        branch1.add_module('quanti{0}'.format(15), Quantization(bit_width, fraction_length[15], is_quantization[15]))

        branch1.add_module('relu8', nn.ReLU(True))
        branch2.add_module('pooling7', nn.MaxPool2d(2, 2))
        branch2.add_module('conv9', nn.Conv2d(nm[7], nm[8], ks[8], ss[8], ps[8]))
        branch2.add_module('quanti{0}'.format(16), Quantization(bit_width, fraction_length[16], is_quantization[16]))

        branch2.add_module('batchnorm9', nn.BatchNorm2d(nm[8]))
        branch2.add_module('quanti{0}'.format(17), Quantization(bit_width, fraction_length[17], is_quantization[17]))

        branch2.add_module('relu9', nn.ReLU(True))

        '''
        fc_branch1 = nn.Sequential()
        fc_branch2 = nn.Sequential()
        fc_branch1. add_module('fc1', nn.Linear(512*1, 76))
        fc_branch1.add_module('quanti{0}'.format(18),Quantization(bit_width, fraction_length[18], is_quantization[18]))

        fc_branch2. add_module('fc2', nn.Linear(512*1, 76))
        fc_branch2.add_module('quanti{0}'.format(19),Quantization(bit_width, fraction_length[19], is_quantization[19]))
        '''

        self.cnn = cnn
        self.branch1 = branch2  # first two character branch
        self.branch2 = branch1  # last five-six character branch
        # self.fc_branch1 = fc_branch1
        # self.fc_branch2 = fc_branch2
        # self.fc2 = nn.Linear(512*1, 76)
        self.fc1 = nn.Linear(512 * 1, 76)
        self.fc2 = nn.Linear(512 * 1, 76)
        self.fc1_quanti = Quantization(bit_width, fraction_length[18], is_quantization[18])
        self.fc2_quanti = Quantization(bit_width, fraction_length[19], is_quantization[19])

    def forward(self, input):
        # conv features
        conv_out = self.cnn(input)

        branch_1 = self.branch1(conv_out)
        b1, c1, h1, w1 = branch_1.size()
        assert h1 == 1, "the height of branch_1 must be 1"
        branch_1 = branch_1.permute(3, 0, 1, 2)  # [w, b, c]
        branch_1 = branch_1.view(branch_1.size(0), branch_1.size(1), -1)
        fc_1 = self.fc1(branch_1)
        fc_1 = self.fc1_quanti(fc_1)

        branch_2 = self.branch2(conv_out)
        b2, c2, h2, w2 = branch_2.size()
        assert h2 == 1, "the height of branch_2 must be 1"
        branch_2 = branch_2.permute(3, 0, 1, 2)  # [w, b, c]
        branch_2 = branch_2.view(branch_2.size(0), branch_2.size(1), -1)
        fc_2 = self.fc2(branch_2)
        fc_2 = self.fc2_quanti(fc_2)
        output = torch.cat((fc_1, fc_2), 0)

        return output


# -------    loader section(custom needed)    -------

class MyDataset(Dataset):
    def __init__(self, label_path, alphabet, resize,
                 img_root='/home/yzzc/Work/lq/carplate_quantize/carplate_recognition/data/test_img'):
        super(MyDataset, self).__init__()
        self.img_root = img_root
        self.labels = self.get_labels(label_path)
        self.alphabet = alphabet
        self.width, self.height = resize

    def __getitem__(self, index):
        image_name = list(self.labels[index].keys())[0]
        path = os.path.join(self.img_root, image_name)

        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        image = cv2.resize(image, (0, 0), fx=self.width / w, fy=self.height / h, interpolation=cv2.INTER_CUBIC)

        image = (numpy.reshape(image, (self.height, self.width, 1))).transpose(2, 0, 1)
        image = self.preprocessing(image)
        return image, index

    def __len__(self):
        return len(self.labels)

    def get_labels(self, label_path):
        # return text labels in a list
        with open(label_path, 'r', encoding='utf-8') as file:
            labels = [{c.strip().split(' ')[0]: c.strip().split(' ')[1]} for c in file.readlines()]

        return labels

    def preprocessing(self, image):
        ## already have been computed
        mean = 0.588
        std = 0.193
        image = image.astype(numpy.float32) / 255.
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image.sub_(mean).div_(std)

        return image


# 加载数据集需要用到的参数
label_txt = '/home/yzzc/Work/lq/carplate_quantize/carplate_recognition/data/quantize_data.txt'
# label_txt = '/home/yzzc/Work/lq/ezai/all_projects/carplate_recognition/data/test_new/split.txt'
img_H = 32
img_W = 100
alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新领学使警挂港澳电"

batch_size = 128

test_data = MyDataset(label_path=label_txt, alphabet=alphabet, resize=(img_W, img_H))
data_loader = DataLoader(dataset=test_data, batch_size=batch_size)


# -------    eval section(custom needed)    -------

def evaluate(model, data_loader):
    model.eval()
    model.cuda()
    accuracy = 0
    n_correct = 0
    with torch.no_grad():
        for i, (image, index) in enumerate(tqdm(data_loader)):
            if torch.cuda.is_available():
                image = image.cuda()
            label = utils.get_batch_label(test_data, index)
            preds = model(image)
            batch_size = image.size(0)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            converter = utils.strLabelConverter(test_data.alphabet)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            raw_preds = converter.decode(preds.data, preds_size.data, raw=True)

            for pred, target in zip(sim_preds, label):
                if pred == target:
                    n_correct += 1

        accuracy = n_correct / float(len(test_data))
    return accuracy


# -------    quantization function section    -------
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


# -------    quantize params    -------
"""
这个部分量化所有的参数，逐层量化，得到最佳量化策略之后测试精度并且保存最佳量化策略下的模型
"""
# for layer in range(len(params)):  # 遍历所有的层数
#     acc_param = 0  # init accuracy
#     print('-- Quantizing layer:{}\'s parameter --'.format(layer))
#     for fraction_length_of_param in range(bit_width):  # 遍历所有的小数位置
#         print('-- Trying fraction length: {} --'.format(fraction_length_of_param))
#
#         for key in params[layer]:  # 量化制定的层
#             param = state[key].clone()  # 提取特定层参数
#             param = Trim2FixedPoint(param.float(), bit_width, fraction_length_of_param)  # 量化
#             state_tmp[key] = param  # 修改临时参数中的指定层参数
#
#         model = Net(bit_width=bit_width,
#                     fraction_length=fraction_length,
#                     is_quantization=is_quantization)  # 模型实例化
#         model.load_state_dict(state_tmp)  # 加载参数
#         acc_param_eval = evaluate(model, data_loader)  # eval
#
#         # #liuqi added
#         # if fraction_length_of_param == 0:
#         #     result_param[layer] = result_param[layer]
#
#         if acc_param_eval >= acc_param:
#             result_param[layer] = [fraction_length_of_param, acc_param_eval]  # 保存小数位置和精度
#         else:
#             result_param[layer] = result_param[layer]
#             # 如果没能获取更高的精度，恢复最好的参数,用于跨层保存最好参数
#             for key_new in params[layer]:
#                 param_new = state[key_new].clone()
#                 param_new = Trim2FixedPoint(param_new.float(), bit_width, result_param[layer][0])
#                 state_tmp[key_new] = param_new
#
#         acc_param = max(acc_param, acc_param_eval)
#
#         print(
#             '-- Accuracy of fraction length: {} is {} --'.format(fraction_length_of_param, acc_param_eval))
#
#     print('-- Layer:{} parameter\'s best result is {} --\n\n'.format(layer, result_param[layer][0]))
# 量化开始前先实例化model
model = Net(bit_width=bit_width, fraction_length=fraction_length, is_quantization=is_quantization)  # 模型实例化
# 使用一个双层循环遍历所有的参数部分
for layer in range(len(params)):
    # 遍历所有的层的所有部分，对于state 字典来说param_name 唯一确定一个参数部分;
    # 对于params result_params 来说二维数组params[layer][index] result_params[layer][index] 唯一确定一个部分
    for index, param_name in enumerate(params[layer]):  # param_name: str 表征一个层的一个参数部分
        acc_max = 0  # init_acc
        print('-- Quantizing {}\'s parameter --'.format(params[layer][index]))
        # 遍历所有的小数位置 fl: fraction_length
        for fl in range(bit_width):
            print('-- Trying fraction length: {} --'.format(fl))
            # 提取特定层的特定部分参数
            param = state[param_name].clone()
            # 量化
            param = Trim2FixedPoint(param.float(), bit_width, fl)
            # 修改tmp参数中的指定层的指定部分参数
            state_tmp[param_name] = param
            # 使用模型加载参数
            model.load_state_dict(state_tmp)
            # 计算精度
            acc_eval = evaluate(model, data_loader)
            # if 精度大于等于初始/上次的精度 ? 替换记录 : 替换tmp上次参数（为了跨层）
            if acc_eval >= acc_max:
                result_param[layer][index] = [param_name, fl, round(acc_eval * 100, 2)]  # 保存小数位置和精度
            else:
                # 获取指定部分参数用作恢复
                param_recover = state[param_name].clone()
                # 记录中是最好的参数，使用最好的参数恢复
                param_recover = Trim2FixedPoint(param_recover.float(), bit_width, result_param[layer][index][1])
                # 把最好的参数装回tmp参数中
                state_tmp[param_name] = param_recover
            # 保证acc_param 一直是最好的精度
            acc_max = max(acc_max, acc_eval)
            print('-- param: {}, fl: {}, acc: {}% --'.format(param_name, fl, round(acc_eval * 100, 2)))
        print('-- param: {}, best_fl: {}, acc_max: {}% --\n\n'
              .format(param_name, result_param[layer][index][1], result_param[layer][index][2]))
final_state = state.copy()
# 使用最佳量化策略，量化预训练模型
# 先遍历层
for layer in result_param:
    # 遍历记录 pr: param_record[param_name, best_fl, acc_max]
    for index, pr in enumerate(layer):
        param = state[pr[0]].clone()
        param = Trim2FixedPoint(param.float(), bit_width, pr[1])
        final_state[pr[0]] = param
model.load_state_dict(final_state)  # eval
acc_eval = evaluate(model, data_loader)  # get eval accuracy
print('-- Quantize parameter is done, best accuracy is {}% --'.format(round(acc_eval * 100, 2)))

# -------    saving quantized model    -------
print("Saving quantized model.")
torch.save(final_state, param_saving_path)  # 保存最佳策略下的参数

# -------    quantize input && output    -------
"""
这个部分是为了获得 fraction_length，这个参数是为模型定义量化的时候准备的
"""
print('-- Start quantizing input and output.')
# 遍历所有层
for layer in range(len(is_quantization)):
    print('-- Quantizing layer:{}\'s inout --'.format(layer))
    acc_max = 0  # init_acc
    # 设置当前层量化输入输出
    is_quantization[layer] = 1
    for fl in range(bit_width):  # 遍历所有的小数位
        print('-- Trying fraction length: {} --'.format(fl))
        fraction_length[layer] = fl
        # 使用当前参数实例化模型
        model = Net(bit_width=bit_width,
                    fraction_length=fraction_length,
                    is_quantization=is_quantization)
        model.load_state_dict(final_state)
        acc_inout_eval = evaluate(model, data_loader)
        # if 精度最佳 ? 保存参数 : 保持不变
        fraction_length[layer] = fl if acc_inout_eval > acc_max else fraction_length[layer]
        acc_max = max(acc_max, acc_inout_eval)

# test section
print('-- Testing --')
is_quantization = numpy.ones_like(fraction_length)  # 返回一个和best一样尺寸的全1矩阵
model = Net(bit_width=bit_width,
            fraction_length=fraction_length,
            is_quantization=is_quantization)
model.load_state_dict(final_state)
acc_inout_eval = evaluate(model, data_loader)
print('-- Quantize inout is done, best accuracy is {} --'.format(acc_inout_eval))

print("Quantization finished!")
