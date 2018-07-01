import torch
import os


class Config:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_name = 'baseline'
    data_dir = '/data1/wurundi/ML/data'
    exp_dir = os.path.join('/data1/wurundi/ML/', exp_name)
    log_dir = os.path.join(exp_dir, 'log/')
    model_dir = os.path.join(exp_dir, 'model/')
    study_type = ['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST']

    def make_dir(self):
        self.exp_dir = os.path.join('/data1/wurundi/ML/', self.exp_name)
        if not os.path.exists(self.exp_dir):
            os.makedirs(os.path.join(self.exp_dir, 'model'))
            os.makedirs(os.path.join(self.exp_dir, 'log'))
        self.log_dir = os.path.join(self.exp_dir, 'log/')
        self.model_dir = os.path.join(self.exp_dir, 'model/')


config = Config()
