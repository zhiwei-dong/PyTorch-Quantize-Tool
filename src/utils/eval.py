# -------    eval section(custom needed)    -------
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.loader import MyDataset
from src.utils import user_utils

# 加载数据集需要用到的参数
label_txt = '/data/zhiwei.dong/datasets/quantize/quantize_data.txt'
# label_txt = '/home/yzzc/Work/lq/ezai/all_projects/carplate_recognition/data/test_new/split.txt'
img_H = 32
img_W = 100
alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新领学使警挂港澳电"
batch_size = 128
test_data = MyDataset(label_path=label_txt, alphabet=alphabet, resize=(img_W, img_H))
data_loader = DataLoader(dataset=test_data, batch_size=batch_size)


def evaluate(model, data_loader, dev='cuda:0'):
    model.eval()
    model.to(dev)
    accuracy = 0
    n_correct = 0
    with torch.no_grad():
        for i, (image, index) in enumerate(tqdm(data_loader)):
            image = image.to(dev)
            label = user_utils.get_batch_label(test_data, index)
            preds = model(image)
            batch_size = image.size(0)
            preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            converter = user_utils.strLabelConverter(test_data.alphabet)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
            for pred, target in zip(sim_preds, label):
                if pred == target:
                    n_correct += 1
        accuracy = n_correct / float(len(test_data))
    return accuracy
