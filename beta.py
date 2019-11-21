from threading import Thread

from main import *


def test_param_parallel(layer, fl, device, acc_max):
    names = locals()
    names['model%s' % fl] = Net(bit_width=bit_width, fraction_length=fraction_length,
                                is_quantization=is_quantization)  # 模型实例化
    for key in params[layer]:  # param_name: str 表征一个层的一个参数部分
        # 提取特定层的特定部分参数
        param = state[key].clone()
        # 量化
        param = float2fixed(param.float(), bit_width, fl)
        # 修改tmp参数中的指定层的指定部分参数
        state_best[key] = param
    # 使用模型加载参数
    names['model%s' % fl].load_state_dict(state_best)
    names['model%s' % fl].to(device)
    # 计算精度
    acc_max[fl] = evaluate(names['model%s' % fl], data_loader, device)


def quantize_param_parallel():
    # -------    quantize params    -------
    """
    这个部分量化所有的参数，逐层量化，得到最佳量化策略之后测试精度并且保存最佳量化策略下的模型
    """
    # 量化开始前先实例化model
    model = Net(bit_width=bit_width, fraction_length=fraction_length, is_quantization=is_quantization)  # 模型实例化
    print('\n-- Starting Quantize parameter. --')
    # 使用一个双层循环遍历所有的参数部分，量化层的一个组合参数而不是单独的参数
    for layer in range(len(params)):
        layer_name = '.'.join(params[layer][0].split('.')[0:-1])
        print('\n-- Quantizing {}\'s parameter --'.format(layer_name))
        # 初始化acc_max数组
        acc_max = [0 for _ in range(bit_width)]
        # 使用多线程来加速，可能会导致负载不均衡，这里是一个demo
        # 使用8个线程4个GPU来加速
        # 遍历所有的小数位置 fl: fraction_length
        threads = []
        # num_dev: cuda设备数目，即可用的显卡数，这些卡被排序为cuda:0 - cuda:num_dev-1
        num_dev = torch.cuda.device_count()
        # 分配
        device = ['cuda:' + str(i % num_dev) for i in range(bit_width)]
        assert len(device) == bit_width, 'Error: program bug.'
        for i, dev in enumerate(device):
            thread = Thread(target=test_param_parallel, args=(layer, i, dev, acc_max))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        # 跨层时使用最好的参数加载到state_best中, 先记录最大值的索引(best_fl)、acc的最大值
        result_param[layer] = [acc_max.index(max(acc_max)), round(max(acc_max), 2)]
        for key in params[layer]:
            param_recover = state[key].clone()
            # 记录中是最好的参数，使用最好的参数恢复
            param_recover = float2fixed(param_recover.float(), bit_width, result_param[layer][0])
            # 把最好的参数装回tmp参数中
            state_best[key] = param_recover
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
    acc_eval = evaluate(model, data_loader)  # get eval accuracy
    print('-- Quantize parameter is done, best accuracy is {}% --\n'.format(round(acc_eval * 100, 2)))

    # -------    saving quantized model    -------
    print("\n-- Saving quantized model to {} --\n".format(param_saving_path))
    torch.save(final_state, param_saving_path)  # 保存最佳策略下的参数
