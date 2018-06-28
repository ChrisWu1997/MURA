import torch
from torch.nn import functional as F
import numpy as np
import cv2
import os
import argparse
from dataset import get_dataloaders
from common import config

def get_cam(features, weights):
    weights = np.reshape(weights, [1, -1, 1, 1])[:,::-1,:,:]
    cam = np.sum(features * weights, axis=1)
    return cam

def main():
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='filepath of the model')
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='mini-batch size')
    args = parser.parse_args()

    net = torch.load(args.model_path)['net']
    net.eval()

    weights = net.module.classifier.weight.data.cpu().numpy()

    net = net.module.features
    dataloader = get_dataloaders('valid', batch_size=args.batch_size, shuffle=True, num_workers=8)

    _, data = next(enumerate(dataloader))
    inputs = data['image']
    labels = data['label']
    ori_filename = data['meta_data']['img_filename']
    inputs = inputs.to(config.device)

    features = net(inputs).cpu().detach().numpy()
    #print('features size:', features.size())
    #heat_map, _ = torch.max(torch.abs(features), dim=1)
    #weights = net.module.classifier.weight.data
    #print('weight size:', weights.size())

    cam = get_cam(features, weights)

    for i in range(inputs.size(0)):
        ori_img = cv2.imread(os.path.join(config.data_dir, ori_filename[i]))
        cv2.imwrite('heatmap/ori_img{}_label{}.jpg'.format(i, labels[i]), ori_img)

        heat_map = cam[i]
        heat_map = np.uint8(heat_map / (np.max(heat_map) - np.min(heat_map))*255)
        heat = cv2.applyColorMap(cv2.resize(heat_map, (ori_img.shape[1], ori_img.shape[0])), cv2.COLORMAP_JET)
        cv2.imwrite('heatmap/heat_map{}_label{}.jpg'.format(i, labels[i]), heat)

        crop = np.uint8(heat_map > 255 * 0.7) * 255
        crop = cv2.resize(crop, (ori_img.shape[1], ori_img.shape[0]))
        cv2.imwrite('heatmap/crop_map{}_label{}.jpg'.format(i, labels[i]), crop)

        result = heat * 0.3 + ori_img * 0.5
        cv2.imwrite('heatmap/result_map{}_label{}.jpg'.format(i, labels[i]), result)

