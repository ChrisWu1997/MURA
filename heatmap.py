import torch
import numpy as np
import argparse
from dataset import get_dataloaders
from common import config
import cv2
import os
import torchvision.transforms as transforms
from tqdm import tqdm
import csv
import time
from PIL import Image

def gray_scale_image(img):
    # Convert a rgb img to gray scale
    grayimg = 0.30*img[:, :, 0] + 0.59*img[:, :, 1] + 0.11*img[:, :, 2]
    return grayimg.astype(np.uint8)



neighbors = [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]


def bfs(i,j,img,map):
    h = img.shape[0]
    w = img.shape[1]
    queue = [(i,j)]
    for ele in queue:
        li,lj = ele[0],ele[1]
        for k in neighbors:
            newi = li + k[0]
            newj = lj + k[1]
            if newi < 0 or newi >= h or newj < 0 or newj >= w or img[newi][newj] == 0 or map[newi][newj] != 0:
                pass
            else:
                map[newi][newj] = map[i][j]
                queue += [(newi,newj)]



def find_max_branch(img):
    h = img.shape[0]
    w = img.shape[1]
    map = np.zeros((h, w))
    branch_cnt = 0
    for i in range(h):
        for j in range (w):
            if img[i][j] == 1 and map[i][j] == 0:
                branch_cnt += 1
                map[i][j] = branch_cnt
                bfs(i,j,img,map)

    branch_size = np.zeros(branch_cnt+1)
    for i in range(h):
        for j in range (w):
            t = int(map[i][j])
            branch_size[t] += 1
    branch_size[0] = 0
    max_branch = np.argmax(branch_size)
    mini = h
    minj = w
    maxi = -1
    maxj = -1
    for i in range(h):
        for j in range(w):
            if map[i][j] == max_branch:
                if i < mini:
                    mini = i
                if i > maxi:
                    maxi = i
                if j < minj:
                    minj = j
                if j > maxj:
                    maxj = j
    return mini,maxi,minj,maxj

def get_local_data_(heat_map,ori_img,i):
    img = heat_map.cpu().detach().numpy()[i]
    img = np.uint8(img / (np.max(img) - np.min(img)) * 255)
    heat = cv2.resize(img, (ori_img.shape[1], ori_img.shape[0]))
    crop = np.uint8(heat > 255 * 0.7)
    mini, maxi, minj, maxj = find_max_branch(crop)
    local_img = ori_img[mini: maxi + 1, minj: maxj + 1, :]
    local_img = cv2.resize(local_img, (224, 224))
    return local_img


def get_center_data_(heat_map,ori_img,i):
    img = heat_map.cpu().detach().numpy()[i]
    img = np.uint8(img / (np.max(img) - np.min(img)) * 255)
    h = ori_img.shape[0]
    w = ori_img.shape[1]
    heat = cv2.resize(img, (ori_img.shape[1], ori_img.shape[0]))
    crop = np.uint8(heat > 255 * 0.7)
    mini, maxi, minj, maxj = find_max_branch(crop)
    ci = int((mini + maxi) / 2)
    cj = int((minj + maxj) / 2)
    mini = max(ci-112,0)
    maxi = min(ci+112,h)
    minj = max(cj-112,0)
    maxj = min(cj+112,w)
    local_img = ori_img[mini: maxi, minj: maxj, :]
    local_img = cv2.resize(local_img, (224, 224))
    return local_img

def get_center_data(heat_map,ori_img,i):
    img = heat_map.cpu().detach().numpy()[i]
    img = np.uint8(img / (np.max(img) - np.min(img)) * 255)
    h = ori_img.shape[0]
    w = ori_img.shape[1]
    heat = cv2.resize(img, (ori_img.shape[1], ori_img.shape[0]))
    crop = np.uint8(heat > 255 * 0.7)
    ret, markers = cv2.connectedComponents(crop)
    branch_size = np.zeros(ret)
    for i in range(h):
        for j in range(w):
            t = int(markers[i][j])
            branch_size[t] += 1
    branch_size[0] = 0
    max_branch = np.argmax(branch_size)
    mini = h
    minj = w
    maxi = -1
    maxj = -1
    for i in range(h):
        for j in range(w):
            if markers[i][j] == max_branch:
                if i < mini:
                    mini = i
                if i > maxi:
                    maxi = i
                if j < minj:
                    minj = j
                if j > maxj:
                    maxj = j
    ci = int((mini + maxi) / 2)
    cj = int((minj + maxj) / 2)
    mini = max(ci - 112, 0)
    maxi = min(ci + 112, h)
    minj = max(cj - 112, 0)
    maxj = min(cj + 112, w)
    local_img = ori_img[mini: maxi, minj: maxj, :]
    local_img = cv2.resize(local_img, (224, 224))
    return local_img

def generate_local(global_features, inputs):
    heat_map, _ = torch.max(torch.abs(global_features), dim=1)
    heat_map = heat_map.cpu().detach().numpy()
    local_img = []
    for i in range(heat_map.shape[0]):
        ori_img = inputs[i]
        local_data = get_local_data(heat_map[i], np.array(ori_img))
        local_img += [local_data]
    local_img = torch.Tensor(np.stack(local_img)).to(config.device)
    return local_img


Trans = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.2056, 0.2056, 0.2056], [0.0313, 0.0313, 0.0313])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_local_data(img,ori_img):
    #ori_img = ori_img.transpose((1,2,0))
    img = np.uint8(img / (np.max(img) - np.min(img)) * 255)
    h = ori_img.shape[0]
    w = ori_img.shape[1]
    heat = cv2.resize(img, (ori_img.shape[1], ori_img.shape[0]))
    crop = np.uint8(heat > 255 * 0.7)
    ret, markers = cv2.connectedComponentsWithStats(crop)
    branch_size = np.zeros(ret)
    for i in range(h):
        for j in range(w):
            t = int(markers[i][j])
            branch_size[t] += 1
    branch_size[0] = 0
    max_branch = np.argmax(branch_size)
    mini = h
    minj = w
    maxi = -1
    maxj = -1
    for i in range(h):
        for j in range(w):
            if markers[i][j] == max_branch:
                if i < mini:
                    mini = i
                if i > maxi:
                    maxi = i
                if j < minj:
                    minj = j
                if j > maxj:
                    maxj = j
    local_img = ori_img[mini: maxi + 1, minj: maxj + 1,: ]

    local_img = Trans(Image.fromarray(local_img))
    '''
    local_img = cv2.resize(local_img, (224, 224))
    local_img = local_img.transpose((2,0,1))
    local_img = torch.from_numpy(local_img)
    print('before normalize')
    print(local_img.size())
    print(torch.max(local_img))
    local_img = transforms.Normalize(mean = [0.485, 0,456, 0.406], std = [0.229, 0.224, 0.225])(local_img)
    '''
    return local_img

def main():
    pass

if __name__ == '__main__':
    main()
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='filepath of the model')
    parser.add_argument('-b', '--batch_size', default=4, type=int, help='mini-batch size')
    parser.add_argument('--phase', default='train', type=str)
    args = parser.parse_args()

    net = torch.load(args.model_path)['net'].module.features
    net.eval()
    dataloader = get_dataloaders(args.phase, batch_size=args.batch_size, shuffle=False, num_workers=16)

    _, data = next(enumerate(dataloader))
    inputs = data['image']
    labels = data['label']
    ori_filename = data['meta_data']['img_filename']
    inputs = inputs.to(config.device)

    with torch.no_grad():
        features = net(inputs)

    generate_local(features)

'''
        heat_map, _ = torch.max(torch.abs(global_features), dim=1)
    for i in range(heat_map.size(0)):
        ori_img = cv2.imread(os.path.join(config.data_dir, ori_filename[i]))
        local_img = get_center_data(heat_map, ori_img, i)
        cv2.imwrite('heatmaps/local{}_label{}.png'.format(i, labels[i]), ori_img)
        cv2.imwrite('heatmaps/ori{}_label{}.png'.format(i, labels[i]), local_img)
'''
