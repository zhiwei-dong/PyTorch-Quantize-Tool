# coding=utf-8

# -------    import section(edit if you need)    -------
import argparse
import os

import numpy

from src.model.model import Net
from src.utils.eval import evaluate, data_loader
from src.utils.function_utils import *
from src.utils.utils import *


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Quantize Tool')
    parser.add_argument('-p', '--pretrain', help='path of pre-trained model',
                        default=
                        '/data/zhiwei.dong/datasets/quantize/best.pth')
    parser.add_argument('-s', '--saving', help='path to saving quantized model',
                        default=
                        './checkpoints/best_quantize.pth')
    parser.add_argument('-m', '--model', help='specific model to quantize',
                        default=
                        'carplate')
    parser.add_argument('-b', '--bit_width', type=int, default=8,
                        help='number of bit you want to quantize pre-trained model (default:8)')
    parser.add_argument('--param_fl', help='path to saving param fl list',
                        default=
                        './checkpoints/best_param.fl')
    parser.add_argument('--inout_fl', help='path to saving inout fl list',
                        default=
                        './checkpoints/best_inout.fl')
    parser.add_argument('--gpu_id', default='3, 4, 5', type=str,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--bn2scale', default=True, type=bool,
                        help='Transfer BN to Scale')
    arguments = parser.parse_args()
    return arguments


def f_eval():
    # 量化开始前先实例化model
    model = Net(bit_width=bit_width, fraction_length=fraction_length, is_quantization=is_quantization)  # 模型实例化
    model.load_state_dict(state)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 开始量化前先测试一下精度
    print('\n-- Starting First Eval. -- ')
    acc = evaluate(model, data_loader, dev)
    print('-- Oringin pre-train model\'s accuracy is {}% --\n'.format(round(acc * 100, 2)))


def test_param(layer, fl, device):
    model = Net(bit_width=bit_width, fraction_length=fraction_length,
                is_quantization=is_quantization)  # 模型实例化
    for key in params[layer]:  # param_name: str 表征一个层的一个参数部分
        # 提取特定层的特定部分参数
        param = state[key].clone()
        # 量化
        param = float2fixed(param.float(), bit_width, fl)
        # 修改tmp参数中的指定层的指定部分参数
        state_best[key] = param
    # 使用模型加载参数
    model.load_state_dict(state_best)
    model.to(device)
    # 计算精度
    acc_eval = evaluate(model, data_loader, device)
    return acc_eval


def quantize_param():
    """
    :param fl:
    :return:
    """
    # 量化开始前先实例化model
    model = Net(bit_width=bit_width, fraction_length=fraction_length, is_quantization=is_quantization)  # 模型实例化
    print('\n-- Starting Quantize parameter. --')
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # 使用一个双层循环遍历所有的参数部分，量化层的一个组合参数而不是单独的参数
    for layer in range(len(params)):
        layer_name = '.'.join(params[layer][0].split('.')[0:-1])
        print('\n-- Quantizing {}\'s parameter --'.format(layer_name))
        acc_max = 0  # init_acc
        # 遍历所有的小数位置 fl: fraction_length
        for fl in range(bit_width):
            print('-- Trying fraction length: {} --'.format(fl))
            acc_eval = test_param(layer, fl, device)
            # if 精度大于等于初始/上次的精度 ? 替换记录 : 替换tmp上次参数（为了跨层）
            if acc_eval >= acc_max:
                result_param[layer] = [fl, round(acc_eval * 100, 2)]  # 保存小数位置和精度
            else:
                # 获取指定部分参数用作恢复
                for key in params[layer]:
                    param_recover = state[key].clone()
                    # 记录中是最好的参数，使用最好的参数恢复
                    param_recover = float2fixed(param_recover.float(), bit_width, result_param[layer][0])
                    # 把最好的参数装回tmp参数中
                    state_best[key] = param_recover
            # 保证acc_param 一直是最好的精度
            acc_max = max(acc_max, acc_eval)
            print('-- layer: {}, fl: {}, acc: {}% --'.format(layer_name, fl, round(acc_eval * 100, 2)))
        print('-- layer: {}, best_fl: {}, acc_max: {}% --\n'
              .format(layer_name, result_param[layer][0], result_param[layer][1]))

    # -------    test section    -------
    final_state = state.copy()
    # 使用最佳量化策略，量化预训练模型
    # 先遍历层
    for index, layer in enumerate(result_param):
        # 遍历记录 layer[best_fl, acc_max]
        for key in params[index]:
            param = state[key].clone()
            param = float2fixed(param.float(), bit_width, layer[0])
            final_state[key] = param
    model.load_state_dict(final_state)  # eval
    acc_eval = evaluate(model, data_loader, device)  # get eval accuracy
    print('-- Quantize parameter is done, best accuracy is {}% --\n'.format(round(acc_eval * 100, 2)))

    # -------    saving quantized model    -------
    print("\n-- Saving quantized model to {} --\n".format(param_saving_path))
    torch.save(final_state, param_saving_path)  # 保存最佳策略下的参数


def quantize_inout(flcheck: list, ) -> None:
    # -------    quantize input && output    -------
    """
    这个部分是为了获得 fraction_length，这个参数是为模型定义量化的时候准备的
    """
    print('\n-- Start quantizing input and output. --')
    # 遍历所有层
    for layer in range(len(is_quantization)):
        layer_name = '.'.join(params[layer][0].split('.')[0:-1])
        print('\n-- Quantizing layer:{}\'s inout --'.format(layer_name))
        acc_max = 0  # init_acc
        # 设置当前层量化输入输出
        is_quantization[layer] = 1
        fl_tmp = fraction_length.copy()
        for fl in range(bit_width):  # 遍历所有的小数位
            print('-- Trying fraction length: {} --'.format(fl))
            fl_tmp[layer] = int(fl)
            # 使用当前参数实例化模型
            model = Net(bit_width=bit_width, fraction_length=fl_tmp, is_quantization=is_quantization)
            model.load_state_dict(state)
            acc_inout_eval = evaluate(model, data_loader)
            # if 精度最佳 ? 保存参数 : 保持不变
            fraction_length[layer] = fl if acc_inout_eval > acc_max else fraction_length[layer]
            acc_max = max(acc_max, acc_inout_eval)
            print('-- layer: {}, fl: {}, acc: {}% --'.format(layer_name, fl, round(acc_inout_eval * 100, 2)))
            save_fl(fraction_length, inout_fl_path)
        print('-- layer: {}, best_fl: {}, acc_max: {}% --\n'
              .format(layer_name, int(fraction_length[layer]), round(acc_max * 100, 2)))

    # -------    test section    -------
    print('\n -- Testing --')
    bool_q = numpy.ones_like(fraction_length)  # 返回一个和best一样尺寸的全1矩阵
    model = Net(bit_width=bit_width, fraction_length=fraction_length, is_quantization=bool_q)
    model.load_state_dict(state)
    acc_inout_eval = evaluate(model, data_loader)
    print('-- Quantize inout is done, best accuracy is {} --\n '.format(acc_inout_eval))


if __name__ == "__main__":
    args = parse_args()
    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    assert torch.cuda.is_available(), 'CUDA is needed for Quantize'
    # -------    global program param    -------
    pretrain_model_path = args.pretrain  # 预训练模型参数路径
    param_fl_path = args.param_fl  # 预训练模型参数路径
    inout_fl_path = args.inout_fl  # 预训练模型参数路径
    param_saving_path = args.saving  # 量化后参数存储的位置
    bit_width = args.bit_width  # 量化的目标比特数
    # map_location 是权宜之计，需要改
    state = torch.load(pretrain_model_path, map_location=torch.device('cpu'))  # 预训练模型参数
    params = get_params(state, args.bn2scale)  # 预训练模型参数列表
    fraction_length = [0 for i in range(len(params))]  # 输入输出量化fl列表
    is_quantization = [0 for i in range(len(params))]  # 是否量化
    state_best = state.copy()  # copy state for test after quantization
    result_param = params.copy()  # copy params structure for record
    # -------    main function    -------
    echo_params(args)
    # f_eval()
    quantize_param()
    # quantize_inout()
    # -------    quantization finished    -------
    print("\n-- Quantization finished! --\n")
