# coding=utf-8

# imports
import torch
import torch.nn as nn

from src.model.overide_layers import Quantization


# -------    model section(custom needed)    -------
class Net(nn.Module):
    """
    # Example for model defination
    class Net(nn.Module):
        def __init__(self, bit_width, fraction_length, is_quantization):
            super(Net, self).__init__()

        def forward(self, x):
            return x
    """

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
