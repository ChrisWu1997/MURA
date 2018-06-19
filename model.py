import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torchvision
from torchvision import transforms, models
import torch.optim as optim
from collections import OrderedDict
from common import config
from tqdm import tqdm
import math
import re
import numpy as np
import cv2

'''
class _BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(_BottleneckBlock, self).__init__()
        inner_channels = out_channels * 4   # some call it bn_size
        self.bn1 = nn.BatchNorm2d(in_channels)  # this is equal to add_module
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inner_channels,
                               kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inner_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, self.drop_rate, training=self.training)
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, self.drop_rate, training=self.training)
        return torch.cat([x, out], dim=1)


class _TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(_TransitionBlock,self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        if self.drop_rate > 0:
            out = F.dropout(out, self.drop_rate, training=self.training)
        out = self.pool(out)
        return out


class _DenseBlock(nn.Module):
    def __init__(self, nr_blocks, in_channels, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        blocks = []
        for i in range(nr_blocks):
            blocks.append(_BottleneckBlock(in_channels+i*growth_rate,
                                          growth_rate, drop_rate))
        self.block = nn.Sequential(*blocks)  # * is used to unpack list

    def forward(self, x):
        return self.block(x)


class DenseNet(nn.Module):
    def __init__(self, block_config=(6, 12, 32, 32), growth_rate=32, drop_rate=0.0, compression=0.5, pretrain=False):
        super(DenseNet, self).__init__()

        init_channels = 2 * growth_rate
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, init_channels, kernel_size=7, stride=2, padding=3,bias=False)),
            ('norm0', nn.BatchNorm2d(init_channels)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))

        nr_channels = init_channels
        for i, nr_block in enumerate(block_config):
            dense_block = _DenseBlock(nr_block, nr_channels, growth_rate, drop_rate)
            self.features.add_module('dense_block{}'.format(i+1), dense_block)
            nr_channels = nr_channels + nr_block * growth_rate
            if i != len(block_config) - 1:
                trans_block = _TransitionBlock(nr_channels, int(nr_channels*compression), drop_rate)
                self.features.add_module('transition_block{}'.format(i+1), trans_block)
                nr_channels = int(nr_channels * compression)

        self.features.add_module('norm5', nn.BatchNorm2d(nr_channels))

        self.linear = nn.Linear(nr_channels, 1)
        #self.linear = nn.Linear(nr_channels, 2)

        if pretrain:
            self.load_state_dict(torch.load(config.pretrain_model), strict=False)
        else:
            # Initialization of Official DenseNet
            for name, param in self.named_parameters():
                if 'conv' in name and 'weight' in name:
                    n = param.size(0) * param.size(2) * param.size(3)
                    param.data.normal_().mul_(math.sqrt(2. / n))
                elif 'norm' in name and 'weight' in name:
                    param.data.fill_(1)
                elif 'norm' in name and 'bias' in name:
                    param.data.fill_(0)
                elif 'classifier' in name and 'bias' in name:
                    param.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = F.sigmoid(self.linear(out))
        #out = F.avg_pool2d(out, kernel_size=1, stride=1).view(features.size(0), -1)
        #out = F.softmax(self.linear(out), dim=1)
        return out
'''

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class network(nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.features = torchvision.models.densenet169(pretrained=True).features
        self.classifier = nn.Linear(1664, 1)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.sigmoid(out)
        return out


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, drop_mode=1):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.drop_mode = drop_mode

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            if self.drop_mode == 1:
                new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """
    def __init__(self, growth_rate=32, block_config=(6, 12, 32, 32),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)

            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.sigmoid(out)
        return out


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            if 'classifier' in key:
                state_dict.pop(key)
                continue
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    return model

def get_activation_map(feature_maps):
    print(feature_maps.size())
    heat_map, _ = torch.max(feature_maps, dim=1).numpy()
    print(heat_map.size())
    for i in range(len(heat_map.size(0))):
        img = heat_map[i]
        img = img / (np.max(img) - np.min(img))
        cv2.imwrite()


def main():
    model = DenseNet()
    print(model)


if __name__ == '__main__':
    #main()
    f = torch.Tensor(np.ones((8, 1024, 11, 11)))
    get_activation_map(f)