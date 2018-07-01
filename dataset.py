from __future__ import print_function, division
import os
import torch
import pandas as pd
import re
#from skimage import io, transform
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
# Ignore warnings
import warnings
from common import config
warnings.filterwarnings("ignore")

#data
class MURA_Dataset(Dataset):
    _patient_re = re.compile(r'patient(\d+)')
    _study_re = re.compile(r'study(\d+)')
    _image_re = re.compile(r'image(\d+)')
    _study_type_re = re.compile(r'XR_(\w+)')

    def __init__(self, data_dir, csv_file, transform=None):
        self.frame = pd.read_csv(os.path.join(data_dir, csv_file), header=None)
        self.data_dir = data_dir
        self.transform = transform
        #self.mean, self.std = self.get_mean_std()
        #self.data_weights = self.get_data_weights()

    def _parse_patient(self, img_filename):
        return int(self._patient_re.search(img_filename).group(1))

    def _parse_study(self, img_filename):
        return int(self._study_re.search(img_filename).group(1))

    def _parse_image(self, img_filename):
        return int(self._image_re.search(img_filename).group(1))

    def _parse_study_type(self, img_filename):
        return self._study_type_re.search(img_filename).group(1)

    def get_data_weights(self):
        img_filenames = [self.frame.iloc[idx,0] for idx in range(self.frame.shape[0])]
        weights = {st:[0.0, 0.0] for st in config.study_type}
        cnt = {st:[0, 0] for st in config.study_type}
        for idx, img_name in enumerate(img_filenames):
            study_type = self._parse_study_type(img_name)
            label = self.frame.iloc[idx, 1]
            cnt[study_type][label] += 1

        for st in config.study_type:
            total = cnt[st][0] + cnt[st][1]
            weights[st][0] = cnt[st][0] / total
            weights[st][1] = cnt[st][1] / total

        return weights

    def get_mean_std(self):
        mean = np.zeros((1,3), dtype=np.float)
        for idx in range(len(self.frame)):
            img_filename = self.frame.iloc[idx, 0]
            image = Image.open(os.path.join(self.data_dir, img_filename)).convert('RGB')
            image = np.array(image) / 255.0
            m = np.mean(image, axis=(0,1))
            mean += np.mean(image, axis=(0, 1))
        mean /= len(self.frame)
        print(mean)

        std = np.zeros((1,3), dtype=np.float)
        for idx in range(len(self.frame)):
            img_filename = self.frame.iloc[idx, 0]
            image = Image.open(os.path.join(self.data_dir, img_filename)).convert('RGB')
            image = np.array(image) / 255.0
            image -= mean
            std += np.mean(image ** 2, axis=(0,1))
        std /= len(self.frame)

        print(mean, std)
        return list(mean), list(std)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self,idx):
        img_filename = self.frame.iloc[idx,0]
        patient = self._parse_patient(img_filename)
        study = self._parse_study(img_filename)
        image_num = self._parse_image(img_filename)
        study_type = self._parse_study_type(img_filename)

        file_path = os.path.join(self.data_dir, img_filename)
        image = Image.open(file_path).convert('RGB')
        #image = cv2.imread(os.path.join(self.data_dir, img_filename))
        label = self.frame.iloc[idx,1]

        meta_data = {
            'y_true': label,
            'img_filename': img_filename,
            'file_path': file_path,
            'patient': patient,
            'study': study,
            'study_type': study_type,
            'image_num': image_num,
            'encounter': "{}_{}_{}".format(study_type, patient, study)
        }

        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label':label, 'meta_data':meta_data}
        return sample


def get_dataloaders(name, batch_size, shuffle, num_workers=32, data_dir=config.data_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            #transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            #transforms.Normalize([0.2056, 0.2056, 0.2056], [0.0313, 0.0313, 0.0313])
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.2056, 0.2056, 0.2056], [0.0313, 0.0313, 0.0313])
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
        ]),
    }

    image_dataset = MURA_Dataset(data_dir=data_dir, csv_file='MURA-v1.0/%s.csv'%(name),
                                     transform=data_transforms[name])

    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    #data_weights = {x : image_dataset[x].data_weights for x in ['train', 'valid']}

    return dataloader


def calc_data_weights():
    frame = pd.read_csv('/data1/wurundi/ML/data/MURA-v1.0/train.csv', header=None)
    N_t = {t:0 for t in config.study_type}
    A_t = {t:0 for t in config.study_type}
    W_t0 = {t:0.0 for t in config.study_type}
    W_t1 = {t:0.0 for t in config.study_type}

    study_type_re = re.compile(r'XR_(\w+)')

    for idx in range(len(frame)):
        img_filename = frame.iloc[idx, 0]
        study_type = study_type_re.search(img_filename).group(1)

        label = frame.iloc[idx, 1]
        if label == 1:
            A_t[study_type] += 1
        else:
            N_t[study_type] += 1

    for t in config.study_type:
        W_t0[t] = A_t[t] / (A_t[t] + N_t[t])
        W_t1[t] = N_t[t] / (A_t[t] + N_t[t])

    return [W_t0, W_t1]


def main():
    calc_data_weights()

if __name__=='__main__':
    main()