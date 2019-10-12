# coding=utf-8
# Program:
# This script is for TG_team to quantize their custom neural network pre-trained model.
# It's a general script to use.
# History:
# 2019/09/27    Albert Dong	First release
# 2019/10/9     Albert Dong	add get model params
# License:
# BSD
##########################################################################

# -------------------------    import section(edit if you need)    -------------------------
import argparse

from tqdm import tqdm
import numpy
import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable

import utils
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------    param section    -------------------------
def get_params(state):
    params = []
    tmplist = []
    tmpname = ''
    # print(state)
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
    params.append(tmplist)
    # print(params)
    return params


"""
Args:
    bit_width 位宽
    fraction_length 小数位长度 仅在输入输出量化时起作用
    is_quantize 是否量化 仅在输入输出量化时起作用
    
"""
parser = argparse.ArgumentParser(description='PyTorch Ristretto Quantization Tool')
parser.add_argument('-p', '--pretrain', help='path of pre-trained model', default='/home/yzzc/Work/lq/license_plate_pytorch/crnn_chinese_characters_rec/expr/all_ft_2/crnn_best.pth')
parser.add_argument('-s', '--saving', help='path to saving quantized model', default='/home/yzzc/Work/lq/license_plate_pytorch/crnn_chinese_characters_rec/expr/all_ft_2/crnn_best_quantize.pth')
parser.add_argument('-b', '--bit_width', type=int, default=8,
                    help='number of bit you want to quantize pretrained model (default:8)')
args = parser.parse_args()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
pretrain_model_path = args.pretrain  # 预训练模型参数路径
param_saving_path = args.saving  # 量化后参数存储的位置
bit_width = args.bit_width  # 量化的目标比特数
state = torch.load(pretrain_model_path)  # 预训练模型参数
params = get_params(state)
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
            cnn.add_module('quanti{0}'.format(2*i),Quantization(bit_width, fraction_length[2*i], is_quantization[2*i]))

            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
                cnn.add_module('quanti{0}'.format(2*i+1),Quantization(bit_width, fraction_length[2*i], is_quantization[2*i]))

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
        branch1.add_module('quanti{0}'.format(12),Quantization(bit_width, fraction_length[12], is_quantization[12]))

        branch1.add_module('batchnorm7', nn.BatchNorm2d(nm[6]))
        branch1.add_module('quanti{0}'.format(13),Quantization(bit_width, fraction_length[13], is_quantization[13]))

        branch1.add_module('relu7', nn.ReLU(True))
        branch1.add_module('conv8',nn.Conv2d(nm[6], nm[7], ks[7], ss[7], ps[7]))
        branch1.add_module('quanti{0}'.format(14),Quantization(bit_width, fraction_length[14], is_quantization[14]))

        branch1.add_module('batchnorm8', nn.BatchNorm2d(nm[7]))
        branch1.add_module('quanti{0}'.format(15),Quantization(bit_width, fraction_length[15], is_quantization[15]))

        branch1.add_module('relu8', nn.ReLU(True))
        branch2.add_module('pooling7',nn.MaxPool2d(2, 2))
        branch2.add_module('conv9',nn.Conv2d(nm[7], nm[8], ks[8], ss[8], ps[8]))
        branch2.add_module('quanti{0}'.format(16),Quantization(bit_width, fraction_length[16], is_quantization[16]))

        branch2.add_module('batchnorm9', nn.BatchNorm2d(nm[8]))
        branch2.add_module('quanti{0}'.format(17),Quantization(bit_width, fraction_length[17], is_quantization[17]))

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
        self.branch1 = branch2 #first two character branch
        self.branch2 = branch1 #last five-six character branch
        # self.fc_branch1 = fc_branch1 
        # self.fc_branch2 = fc_branch2 
        # self.fc2 = nn.Linear(512*1, 76)
        self.fc1 = nn.Linear(512*1, 76)
        self.fc2 = nn.Linear(512*1, 76)
        self.fc1_quanti = Quantization(bit_width, fraction_length[18], is_quantization[18])
        self.fc2_quanti = Quantization(bit_width, fraction_length[19], is_quantization[19])
        

    def forward(self, input):
        # conv features
        conv_out = self.cnn(input)
        
        branch_1 = self.branch1(conv_out)
        b1, c1, h1, w1 = branch_1.size()
        assert h1 == 1, "the height of branch_1 must be 1"
        branch_1 = branch_1.permute(3,0,1,2)  # [w, b, c]
        branch_1 = branch_1.view(branch_1.size(0),branch_1.size(1),-1)
        fc_1 = self.fc1(branch_1)
        fc_1 = self.fc1_quanti(fc_1)

        branch_2 = self.branch2(conv_out)
        b2, c2, h2, w2 = branch_2.size()
        assert h2 == 1, "the height of branch_2 must be 1"
        branch_2 = branch_2.permute(3,0,1,2)  # [w, b, c]
        branch_2 = branch_2.view(branch_2.size(0),branch_2.size(1),-1)
        fc_2 = self.fc2(branch_2)
        fc_2 = self.fc2_quanti(fc_2)
        output = torch.cat((fc_1,fc_2), 0)

        return output

# -------------------------    loader section(custom needed)    -------------------------

class MyDataset(Dataset):
    def __init__(self, label_path, alphabet, resize, img_root=''):
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
        image = cv2.resize(image, (0,0), fx=self.width/w, fy=self.height/h, interpolation=cv2.INTER_CUBIC)

        image = (numpy.reshape(image, (self.height, self.width, 1))).transpose(2, 0, 1)
        image = self.preprocessing(image)
        return image, index


    def __len__(self):
        return len(self.labels)


    def get_labels(self, label_path):
        # return text labels in a list
        with open(label_path, 'r', encoding='utf-8') as file:
            labels = [ {c.strip().split(' ')[0]:c.strip().split(' ')[1]}for c in file.readlines()]  

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


# -------------------------    eval section(custom needed)    -------------------------

def evaluate(model, data_loader, max_i=1000):
    print('Start val')
    model = model.to(device)
    model.eval()
    model.cuda()
    accuracy = 0
    n_correct = 0
    with torch.no_grad():
        for i, (image, index) in enumerate(data_loader):
            if i%10==0:
                print(i)
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
                # print(i, pred, target)
                if pred == target:
                    n_correct += 1

        accuracy = n_correct / float(len(test_data))####
        print(n_correct, len(test_data))
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
            # print('before: ', param)
            param = Trim2FixedPoint(param.float(), bit_width, fraction_length_of_param)  # 量化
            # print('after: ', param)

            state_tmp[key] = param  # 修改临时参数中的指定层参数

        model = Net(bit_width=bit_width,
                    fraction_length=fraction_length,
                    is_quantization=is_quantization)  # 模型实例化
        model.load_state_dict(state_tmp)  # 加载参数
        acc_param_eval = evaluate(model, data_loader)  # eval

        # #liuqi added
        # if fraction_length_of_param == 0:
        #     result_param[layer] = result_param[layer]

        if acc_param_eval >= acc_param:
            result_param[layer] = [fraction_length_of_param, acc_param_eval]  # 保存小数位置和精度
        else:
            result_param[layer] = result_param[layer]
            # 如果没能获取更高的精度，恢复最好的参数,用于跨层保存最好参数
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
