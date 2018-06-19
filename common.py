import torch
import os
import json


class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = 'baseline'
    data_dir = '/data1/wurundi/ML/data'
    exp_dir = os.path.join('/data1/wurundi/ML/', exp_name)
    log_dir = os.path.join(exp_dir, 'log/')
    model_dir = os.path.join(exp_dir, 'model/')
    pretrain_model = '/data1/wurundi/ML/model/densenet169-b2777c0a.pth'
    study_type = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']

    def make_dir(self):
        self.exp_dir = os.path.join('/data1/wurundi/ML/', self.exp_name)
        if os.path.exists(self.exp_dir) == False:
            os.makedirs(os.path.join(self.exp_dir, 'model'))
            os.makedirs(os.path.join(self.exp_dir, 'log'))
        self.log_dir = os.path.join(self.exp_dir, 'log/')
        self.model_dir = os.path.join(self.exp_dir, 'model/')

config = Config()