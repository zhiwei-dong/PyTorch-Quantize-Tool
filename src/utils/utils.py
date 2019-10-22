# coding=utf-8

# imports
import torch


# -------    quantization function section    -------
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


def float2fixed(data, bit_width=8, fraction_length=0):
    """
    :parameter
        data 原数据
        bit_width 位宽
        fraction_length 小数部分长度
    """
    max_data = (pow(2, bit_width - 1) - 1) * pow(2, -fraction_length)
    # 当前设定位宽和fl的最大值
    min_data = -pow(2, bit_width - 1) * pow(2, -fraction_length)
    # 当前设定位宽和fl的最小值
    # https://pytorch.org/docs/stable/torch.html?highlight=clamp#torch.clamp
    data = torch.clamp(data, min_data, max_data)  # 设置上下极限，使得原数据被限制在本动态范围内
    data /= pow(2, -fraction_length)  # 除分数部分
    data = torch.floor(data)  # 得到小的最接近的整数d
    data[data % 2 != 0] += 1  # ？
    data *= pow(2, -fraction_length)  # 乘回去
    return data
