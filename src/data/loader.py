# coding=utf-8

# imports
import os

import cv2
import numpy
import torch
from torch.utils.data import Dataset


# -------    loader section(custom needed)    -------
class MyDataset(Dataset):
    def __init__(self, label_path, alphabet, resize,
                 img_root=''):
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
