# coding=utf-8

# imports
import json


def load_fl(path: str) -> list:
    """
    This function use json load best or checkpoint fl.
    :param path:
    :return:
    """
    with open(path, 'r') as file:
        fl = json.load(file)
        return fl


def save_fl(fl: list, path: str, blist: list = None) -> str:
    """
    This function use json module save fl to disk.
    :param blist:
    :param fl: fl array generated by quantization function
    :param path: path for fl to save.
    :return:
    """
    with open(path, 'w') as file:
        json.dump(fl, file)
    return path


def echo_params(function_param):
    """
    :param function_param: sys.argv
    :return: None
    """
    print('\n-- Program Params -- ')
    print("gpu id: " + str(function_param.gpu_id))
    print("bit width: " + str(function_param.bit_width))
    print("saving path: " + str(function_param.saving))
    print("pretrain path: " + str(function_param.pretrain))
    print("transfer BN to Scale: " + str(function_param.bn2scale))
