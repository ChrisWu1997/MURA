import torch
import numpy as np
from model import get_activation_map
import argparse
from dataset import get_dataloaders
from common import config
import cv2
import os

def main():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='filepath of the model')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='mini-batch size')
    args = parser.parse_args()

    net = torch.load(args.model_path)['net']
    net.eval()
    net = net.module.features
    dataloader = get_dataloaders('valid', batch_size=args.batch_size, shuffle=True, num_workers=8)

    _, data = next(enumerate(dataloader))
    inputs = data['image']
    labels = data['label']
    ori_filename = data['meta_data']['img_filename']
    inputs = inputs.to(config.device)

    features = net(inputs)

    heat_map, _ = torch.max(torch.abs(features), dim=1)

    for i in range(heat_map.size(0)):
        ori_img = cv2.imread(os.path.join(config.data_dir, ori_filename[i]))
        cv2.imwrite('heatmap/ori_img{}_label{}.jpg'.format(i, labels[i]), ori_img)

        img = heat_map.cpu().detach().numpy()[i]
        img = np.uint8(img / (np.max(img) - np.min(img))*255)
        #img = cv2.resize(img, (224, 224))
        #img = np.reshape(img, (1, 224, 224))
        #print(img.shape)
        #cv2.imwrite('heatmap/heat_map{}.png'.format(i), img)
        heat = cv2.applyColorMap(cv2.resize(img, (ori_img.shape[1], ori_img.shape[0])), cv2.COLORMAP_JET)
        cv2.imwrite('heatmap/heat_map{}_label{}.jpg'.format(i, labels[i]), heat)

        crop = np.uint8(img > 255 * 0.7) * 255
        crop = cv2.resize(crop, (ori_img.shape[1], ori_img.shape[0]))
        cv2.imwrite('heatmap/crop_map{}_label{}.jpg'.format(i, labels[i]), crop)

        result = heat * 0.3 + ori_img * 0.5
        cv2.imwrite('heatmap/result_map{}_label{}.jpg'.format(i, labels[i]), result)
