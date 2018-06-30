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
#from misc_functions import get_params, save_class_activation_on_image

Trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
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
