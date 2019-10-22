# coding=utf-8

# imports
import torch.nn as nn

from src.utils.utils import float2fixed


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
            tmp = float2fixed(data, bit_width=self.bit_width, fraction_length=self.fl)
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
            tmp = float2fixed(data, bit_width=self.bit_width, fraction_length=self.fl)
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
            new_data = float2fixed(data, bit_width=self.bit_width, fraction_length=self.fraction_length)
            return new_data
