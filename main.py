"""
Created on Wed Feb 7 2018
main file
@author: mengshu
"""
from train import TrainModel
import os

"""
DON'T CHANGE THE CUR_DIR
"""
cur_dir = os.getcwd()

train = TrainModel(cur_dir,  suffix = '.tif', size = 512)
train.run()
