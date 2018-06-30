import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse
from PIL import Image
import os
import time
from dataset import get_dataloaders

Trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.2056, 0.2056, 0.2056], [0.0313, 0.0313, 0.0313])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

Normal = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.2056, 0.2056, 0.2056], [0.0313, 0.0313, 0.0313])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def generate_grad_cam(net, input_image):
    """
    generate gradient class activation map
    :param net: deep learning network, supposed to be ResNet
    :param input_image: input to the net, transformed by Trans
    :return: gradient class activation map
    """

    feature = None
    gradient = None

    def func_f(module, input, output):
        nonlocal feature
        feature = output.data.cpu().numpy()

    def func_b(module, grad_in, grad_out):
        nonlocal gradient
        gradient = grad_out[0].data.cpu().numpy()

    net.module.layer4.register_forward_hook(func_f)
    net.module.layer4.register_backward_hook(func_b)

    out = net(input_image.unsqueeze(0))

    pred = (out.data > 0.5)

    net.zero_grad()

    loss = F.binary_cross_entropy(out, pred.float())
    loss.backward()

    feature = np.squeeze(feature, axis=0)
    gradient = np.squeeze(gradient, axis=0)

    weights = np.mean(gradient, axis=(1, 2), keepdims=True)
    #weights = net.module.fc.weight.data.cpu().numpy().reshape((-1, 1, 1))

    cam = np.sum(weights * feature, axis=0)

    cam = cv2.resize(cam, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
    cam = 1.0 - cam
    cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize

    return cam


def localize(cam_feature, input_image):
    """
    localize the abnormality region using grad_cam feature
    :param cam_feature: cam_feature by enerate_grad_cam
    :param input_image: input of the network
    :return: img with heatmap, the abnormality region is highlighted
    """
    activation_heatmap = cv2.applyColorMap(cam_feature, cv2.COLORMAP_JET)

    ori_img = invTrans(input_image).numpy() * 255
    ori_img = np.transpose(ori_img, (1, 2, 0))

    img_with_heatmap = 0.15 * np.float32(activation_heatmap) + 0.85 * np.float32(ori_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap) * 255

    return img_with_heatmap


def generate_local(cam_features, inputs):
    """
    :param cam_features: numpy array of shape = (B, 224, 224), pixel value range [0, 255]
    :param inputs: tensor of size = (B, 3, 224, 224), with mean and std as Imagenet
    :return: local image
    """
    b = cam_features.shape[0]
    local_out = []
    for k in range(b):
        ori_img = invTrans(inputs[k]).cpu().numpy()
        ori_img = np.transpose(ori_img, (1, 2, 0))
        ori_img = np.uint8(ori_img * 255)
        #cv2.imwrite('localize_result/ori' + str(k) + '.png', ori_img)

        #if k==0:
        #    cv2.imwrite("ori_img.png", ori_img)
        crop = np.uint8(cam_features[k] > 0.7)
        ret, markers = cv2.connectedComponents(crop)
        branch_size = np.zeros(ret)
        h = 224
        w = 224
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
        local_img = ori_img[mini: maxi + 1, minj: maxj + 1, :]
        #print('local shape', local_img.shape)
        local_img = cv2.resize(local_img, (224, 224))
        #cv2.imwrite('localize_result/local' + str(k) + '.png', local_img)
        local_img = Image.fromarray(local_img)
        local_img = Normal(local_img)
        local_out += [local_img]
    #local_out = np.array(local_out)
    local_out = torch.stack(local_out)
    #print('local out:', local_out.size())
    return local_out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='filepath of the model')
    parser.add_argument('--img_path', type=str, required=False, help='filepath of query input')
    parser.add_argument('--save_path', type=str, required=False, help='filepath of output heatmap')
    args = parser.parse_args()

    net = torch.load(args.model_path)['net']

    since = time.time()

    ori_image = Image.open(args.img_path).convert('RGB')
    input_image = Trans(ori_image)

    cam_feature = generate_grad_cam(net, input_image)
    result = localize(cam_feature, input_image)

    cv2.imwrite(args.save_path, result)

    print('total time:', time.time() - since)
