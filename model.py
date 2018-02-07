#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 2018
Training pytorch model here
@author: mengshu
"""

from __future__ import print_function
import numpy as np  
from os.path import exists, join
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, ToTensor
from os import listdir
import os
from PIL import Image 
import torchvision
from math import sqrt
import sys


class UNet(nn.Module):
    def __init__(self, colordim):
        super(UNet, self).__init__()
        """
        private parameters
        including the structure of the network
        """
        
        self.__conv1_1 = nn.Conv2d(colordim, 64, 3, padding = 1)
        self.__conv1_2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.__bn1_1 = nn.BatchNorm2d(64)
        self.__bn1_2 = nn.BatchNorm2d(64)
        self.__conv2_1 = nn.Conv2d(64, 128, 3, padding = 1)
        self.__conv2_2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.__bn2_1 = nn.BatchNorm2d(128)
        self.__bn2_2 = nn.BatchNorm2d(128)
        self.__conv4_1 = nn.Conv2d(128, 256, 3, padding = 1)
        self.__conv4_2 = nn.Conv2d(256, 256, 3, padding = 1)
        self.__upconv4 = nn.Conv2d(256, 128, 1)
        self.__bn4 = nn.BatchNorm2d(128)
        self.__bn4_1 = nn.BatchNorm2d(256)
        self.__bn4_2 = nn.BatchNorm2d(256)
        self.__bn4_out = nn.BatchNorm2d(256)
        self.__conv7_1 = nn.Conv2d(256, 128, 3, padding = 1)
        self.__conv7_2 = nn.Conv2d(128, 128, 3, padding = 1)
        self.__upconv7 = nn.Conv2d(128, 64, 1)
        self.__bn7 = nn.BatchNorm2d(64)
        self.__bn7_1 = nn.BatchNorm2d(128)
        self.__bn7_2 = nn.BatchNorm2d(128)
        self.__bn7_out = nn.BatchNorm2d(128)
        self.__conv9_1 = nn.Conv2d(128, 64, 3, padding = 1)
        self.__conv9_2 = nn.Conv2d(64, 64, 3, padding = 1)
        self.__bn9_1 = nn.BatchNorm2d(64)
        self.__bn9_2 = nn.BatchNorm2d(64)
        self.__conv9_3 = nn.Conv2d(64, colordim, 1)
        self.__bn9_3 = nn.BatchNorm2d(colordim)
        self.__bn9 = nn.BatchNorm2d(colordim)
        self.__maxpool = nn.MaxPool2d(2, stride = 2, return_indices = False, ceil_mode = False)
        self.__upsample = nn.UpsamplingBilinear2d(scale_factor = 2)
        self._initialize_weights()

    
    def forward(self, x1):
        
        x1 = F.relu(self.__bn1_2(self.__conv1_2(F.relu(self.__bn1_1(self.__conv1_1(x1))))))
        x2 = F.relu(self.__bn2_2(self.__conv2_2(F.relu(self.__bn2_1(self.__conv2_1(self.__maxpool(x1)))))))
        xup = F.relu(self.__bn4_2(self.__conv4_2(F.relu(self.__bn4_1(self.__conv4_1(self.__maxpool(x2)))))))
        xup = self.__bn4(self.__upconv4(self.__upsample(xup)))
        xup = self.__bn4_out(torch.cat((x2, xup), 1))
        xup = F.relu(self.__bn7_2(self.__conv7_2(F.relu(self.__bn7_1(self.__conv7_1(xup))))))
        xup = self.__bn7(self.__upconv7(self.__upsample(xup)))
        xup = self.__bn7_out(torch.cat((x1, xup), 1))
        xup = F.relu(self.__bn9_3(self.__conv9_3(F.relu(self.__bn9_2(self.__conv9_2(F.relu(self.__bn9_1(self.__conv9_1(xup))))))))) 
        
        return F.softsign(self.__bn9(xup))
    
    
    def _initialize_weights(self):
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_() 





class TrainModel():
    

    def __init__(self, cur_dir, suffix = '.tif', cuda = True, testBatchSize = 4, 
                 batchSize = 4, nEpochs = 200, lr = 0.001, threads = 4, 
                 seed = 123, size = 256, 
                 input_transform = True, target_transform = True):
#        super(TrainModel, self).__init__()
        
        self.data_dir = cur_dir + '/data/'
        self.suffix = suffix
        """
        training parameters are set here
        
        """
        self.colordim = 1
        self.cuda = cuda
        if self.cuda and not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")
        self.testBatchSize = testBatchSize
        self.batchSize = batchSize
        self.nEpochs = nEpochs
        self.lr = lr
        self.threads = threads
        self.seed = seed
        self.size = size

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.__check_dir = cur_dir + '/checkpoint'
        if not exists(self.__check_dir):
            os.mkdir(self.__check_dir)
        self.__epoch_dir = cur_dir + '/epoch'
        if not exists(self.__epoch_dir):
            os.mkdir(self.__epoch_dir)
        
        """
        initialize the model
        """
        
        
        if self.cuda:
            self.unet = UNet(self.colordim).cuda()
            self.criterion = nn.MSELoss().cuda()
        else:
            self.unet = UNet(self.colordim)
            self.criterion = nn.MSELoss()
        
        self.optimizer = optim.SGD(self.unet.parameters(), lr = self.lr)

        
    def __dir_exist(self, cur_dir):
        if not exists(cur_dir):
            sys.exit(cur_dir +' does not exist...')
        return cur_dir
        """
        need to be completed later
        add some other functions such as function for checking if there are 
        train and test subfolders in the directory
        """
    def __get_training_set(self):
        root_dir = self.__dir_exist(self.data_dir)
        train_dir = self.__dir_exist(join(root_dir, "train"))
        return DatasetFromFolder(train_dir,  
                                 colordim = self.colordim, size = self.size,
							          _input_transform = self.input_transform,
							          _target_transform = self.target_transform,
                                      suffix = self.suffix)   
    def __get_test_set(self):
        root_dir = self.__dir_exist(self.data_dir)
        test_dir = self.__dir_exist(join(root_dir, "test"))
        return DatasetFromFolder(test_dir, 
                                 colordim = self.colordim, size = self.size,
							          _input_transform = self.input_transform,
							          _target_transform = self.target_transform,
                                      suffix = self.suffix)  
        
    def __get_ready_for_data(self):
        
        self.train_set = self.__get_training_set()
        self.test_set = self.__get_test_set()
        self.training_data_loader = DataLoader(dataset = self.train_set, 
        								  num_workers = self.threads,
        								  batch_size = self.batchSize,
        								  shuffle = True)
        self.testing_data_loader = DataLoader(dataset = self.test_set, 
        								 num_workers = self.threads,
        								 batch_size = self.testBatchSize,
        								 shuffle = False)
        
    def __train(self, epoch):
        
        epoch_loss = 0
        
        for iteration, (batch_x, batch_y) in enumerate(self.training_data_loader):
            
            input  = Variable(batch_x)
            target = Variable(batch_y)
            if self.cuda:
                input  = input.cuda()
                target = target.cuda()
            
            input  = self.unet(input)
            loss = self.criterion(input, target)
            epoch_loss += (loss.data[0])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            if iteration % 50 is 0:
                print("===> Epoch[{}]({}/{}) : Loss: {:.4f}".format(epoch, iteration, len(self.training_data_loader), loss.data[0]))
        
        result1 = input.cuda()
        imgout = torch.cat([target, result1], 2)
        torchvision.utils.save_image(imgout.data, self.__epoch_dir +'/'+ str(epoch) + self.suffix)
        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(self.training_data_loader)))
        
        return epoch_loss/len(self.training_data_loader)

    def __test(self):
        
        totalloss = 0
        for batch in self.testing_data_loader:
            
            input = Variable(batch[0], volatile = True)
            target =  Variable(batch[1][:, :, :, :],volatile = True)
            
            if self.cuda:
                input = input.cuda()
                target = target.cuda()
            self.optimizer.zero_grad()
            prediction = self.unet(input)
            loss = self.criterion(prediction, target)
            totalloss += loss.data[0]
            
        print("===> Avg. test loss: {:,.4f} dB".format(totalloss / len(self.testing_data_loader)))
        
    def __checkpoint(self, epoch):
        
        model_out_path = (self.__check_dir + "/model_epoch_{}.pth").format(epoch)
        torch.save(self.unet.state_dict(), model_out_path)
#        self.__print("Checkpoint saved to {}.".format(model_out_path))
        
    def run(self):
        
        if self.cuda:
            torch.cuda.manual_seed(self.seed)
        else:
            torch.manual_seed(self.seed)    
        
        print("===> Loading data")
        self.__get_ready_for_data()
        
        print("===> Building unet")
        print("===> Training unet")
        
        
        for epoch in range(1, self.nEpochs + 1):
            
            avg_loss = self.__train(epoch)
            
            
            if epoch % 20 is 0:
                self.__checkpoint(epoch)
                self.__test()
                
        if not self.__exit:
            self.__checkpoint(epoch)


    def use_model_on_one_image(self, image_path, model_path, save_path):
        """
        use an existed model on one image
        """
        if self.cuda:
            self.unet.load_state_dict(torch.load(model_path))
        else:
            self.unet.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))

        ori_image = Image.open(image_path).convert('L')
        transform = ToTensor()

        input = transform(ori_image)
        if self.cuda:
            input = Variable(input.cuda())
        else:
            input = Variable(input)
        input = torch.squeeze(input,0)

        output = unet(input)

        if self.cuda:
            output = output.cuda()

        result = torch.cat([input.data, output.data], 0)

        torchvision.utils.save_image(result, save_path)




