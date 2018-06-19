from __future__ import print_function, division
import os
import torch
import pandas as pd
import re
#from skimage import io, transform
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
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

    def __init__(self, data_dir, csv_file, type, transform=None):
        self.frame_all = pd.read_csv(os.path.join(data_dir, csv_file), header=None)
        self.iloc = []
        total = len(self.frame_all)
        st = total
        for i in range(total):
            filename = self.frame_all.iloc[i,0]
            if self._parse_study_type(filename)==type:
                st = i
                break

        for i in range(st,total):
            filename = self.frame_all.iloc[i, 0]
            if self._parse_study_type(filename)!=type:
                break
            self.iloc += [[filename,self.frame_all.iloc[i,1]]]

        self.data_dir = data_dir
        self.transform = transform
        self.type = type
        self.data_weights = self.get_data_weights()

    def _parse_patient(self, img_filename):
        return int(self._patient_re.search(img_filename).group(1))

    def _parse_study(self, img_filename):
        return int(self._study_re.search(img_filename).group(1))

    def _parse_image(self, img_filename):
        return int(self._image_re.search(img_filename).group(1))

    def _parse_study_type(self, img_filename):
        return self._study_type_re.search(img_filename).group(1)

    def get_data_weights(self):
        img_filenames = [self.iloc[idx][0] for idx in range(len(self.iloc))]
        # balance positive and negative samples
        count = [0, 0]
        total = len(img_filenames)
        for idx, img_name in enumerate(img_filenames):
            label = self.iloc[idx][1]
            count[label] += 1

        weights = [total/float(count[0]), total/float(count[1])]
        sample_weights = [0] * len(img_filenames)
        for idx, img_name in enumerate(img_filenames):
            label = self.iloc[idx][1]
            sample_weights[idx] = weights[label]

        return sample_weights

    def __len__(self):
        return len(self.iloc)

    def __getitem__(self,idx):
        img_filename = self.iloc[idx][0]
        patient = self._parse_patient(img_filename)
        study = self._parse_study(img_filename)
        image_num = self._parse_image(img_filename)
        study_type = self._parse_study_type(img_filename)

        image = Image.open(os.path.join(self.data_dir, img_filename)).convert('RGB')
        label = self.iloc[idx][1]

        meta_data = {
            'y_true': label,
            'img_filename': img_filename,
            'patient': patient,
            'study': study,
            'study_type': study_type,
            'image_num': image_num,
            'encounter': "{}_{}_{}".format(study_type, patient,study)
        }

        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'label':label, 'meta_data':meta_data}
        return sample


def get_dataloaders(name, batch_size, shuffle, type):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224),
            #transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
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

    image_dataset = MURA_Dataset(data_dir=config.data_dir, csv_file='MURA-v1.0/%s.csv'%(name),
                                 type=type, transform=data_transforms[name])

    dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    #data_weights = {x : image_dataset[x].data_weights for x in ['train', 'valid']}

    return dataloader

'''
def get_dataloaders(batch_size,type):
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.RandomResizedCrop(224),
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((224,224)),
            #transforms.CenterCrop(224),
            #transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_dataset = {x: MURA_Dataset(data_dir=config.data_dir + '/', csv_file='MURA-v1.0/%s.csv' % (x), type=type,
                                     transform=data_transforms[x]) for x in ['train', 'valid']}
    
    #train_sample_weights = image_dataset['train'].data_weights
    #train_sample_weights = torch.DoubleTensor(train_sample_weights)
    #sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
    

    train_loader = DataLoader(image_dataset['train'], batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(image_dataset['valid'], batch_size=batch_size, shuffle=False, num_workers=8)

    dataloaders = {'train':train_loader, 'valid':val_loader}

    return dataloaders
'''

if __name__ == '__main__':
    dataloaders = get_dataloaders(1, 'ELBOW')
    print(len(dataloaders['train']), len(dataloaders['valid']))
