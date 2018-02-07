"""
Created on Wed Feb 7 2018
Load training and testing dataset
@author: mengshu
"""

from __future__ import print_function
import numpy as np
from os.path import exists, join
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from os import listdir
import os
from PIL import Image
import sys

class DatasetFromFolder(data.Dataset):

    def __init__(self, image_dir, colordim, size, _input_transform = False, _target_transform = False, suffix = '.tif'):

        super(DatasetFromFolder, self).__init__()
        self.suffix = suffix
        self.mask_filenames = [x for x in listdir(image_dir) if any(x.endswith(extension) for extension in ['_mask'+self.suffix])]
        self._input_transform = _input_transform
        self._target_transform = _target_transform
        self.image_dir = image_dir

        self.colordim = colordim
        self.size = size



    def load_img(self, filepath):

        if self.colordim == 1:
            img = Image.open(filepath).convert('L')
        else:
            img = Image.open(filepath).convert('RGB')
        return img


    def __getitem__(self, index):

        data_suffix = self.suffix
        mask_suffix = '_mask' + self.suffix
        mask_name = join(self.image_dir, self.mask_filenames[index])
        image_name = mask_name.replace(mask_suffix, data_suffix)
        input =  self.load_img(image_name)
        target = self.load_img(mask_name)
        temp = np.fromiter(iter(target.getdata()), np.uint8)
        temp.resize(target.height, target.width)
        target = Image.fromarray(temp, mode = 'L')

        if self._input_transform:
            transform = Compose([CenterCrop(self.size), ToTensor()])
            input = transform(input)
        if self._target_transform:
            transform = Compose([CenterCrop(self.size), ToTensor()])
            target = transform(target)
        return input, target


    def __len__(self):

        return len(self.mask_filenames)
