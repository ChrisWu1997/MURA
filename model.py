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
from heatmap import generate_local

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
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
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
            else:
                new_features = F.dropout2d(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, drop_mode=1):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, drop_mode)
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
            if i == 0:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, drop_mode=2)
            else:
                block = _DenseBlock(num_layers=num_layers, num_input_features=num_features,
                                    bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, drop_mode=1)

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

    def forward(self, x, required_feature=False):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        out = F.sigmoid(out)
        if required_feature:
            return out, features
        else:
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




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        #self.drop_mode = drop_mode
        #self.drop_rate = drop_rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        '''
        if self.drop_rate > 0:
            if self.drop_mode == 1:
                out = F.dropout(out, p=self.drop_rate, training=self.training)
            else:
                out = F.dropout2d(out, p=self.drop_rate, training=self.training)
        out = self.bn2(out)
        '''

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        #self.drop_mode = drop_mode
        #self.drop_rate = drop_rate

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        '''
        if self.drop_rate > 0:
            if self.drop_mode == 1:
                out = F.dropout(out, p=self.drop_rate, training=self.training)
            else:
                out = F.dropout2d(out, p=self.drop_rate, training=self.training)
        '''
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, drop_rate=0):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])#, drop_mode=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#, drop_mode=1, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)#, drop_mode=1, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)#, drop_mode=1, drop_rate=drop_rate)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, required_feature=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if required_feature:
            return x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.sigmoid(x)
        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        #model.load_state_dict(model_zoo.load_url(model_urls['resnet50']), strict=False)
        state_dict = model_zoo.load_url(model_urls['resnet50'])
        #print(list(state_dict.keys()))
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet101'])
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        model.load_state_dict(state_dict, strict=False)
    return model

'''
def local_gen_(global_features, ori_inputs):
    # ----- parameters
    # global features: Tensor of size B * C * 7 * 7
    # ori_inputs B * 3 * H * W
    # ----- return value
    # local_inputs: Tensor of size B * 3 * 224 * 224
    #               with mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    #               (this can be done transforms.Normalize(), see dataset.py)
    pass
'''

GLOBAL_BRANCH_DIR = '/data1/wurundi/ML/baseline5_resnet50/model/best_model.pth.tar'
LOCAL_BRANCH_DIR = '/data1/wurundi/ML/baseline5_resnet50_local/model/best_model.pth.tar'

class fusenet(nn.Module):
    def __init__(self, global_branch=None, local_branch=None, num_features=2048):
        super(fusenet, self).__init__()
        self.global_branch = resnet50()
        self.global_branch.load_state_dict(global_branch)
        self.local_branch = resnet50()
        self.local_branch.load_state_dict(local_branch)
        #self.local_gen = local_gen
        self.classifier = nn.Linear(2*num_features, 1)

    def forward(self, x, ori_filepaths):
        g_features = self.global_branch(x, required_feature=True)

        local_x = generate_local(g_features, ori_filepaths)
        #g_features = F.relu(g_features, inplace=True)
        g_features = F.avg_pool2d(g_features, kernel_size=7, stride=1).view(g_features.size(0), -1)

        l_features = self.local_branch(local_x, required_feature=True)
        #l_features = F.relu(l_features, inplace=True)
        l_features = F.avg_pool2d(l_features, kernel_size=7, stride=1).view(l_features.size(0), -1)

        out = self.classifier(torch.cat([g_features, l_features], 1))
        out = F.sigmoid(out)
        return out



def main():
    '''
    global_branch = torch.load(GLOBAL_BRANCH_DIR)['net'].module.children()

    #net = fusenet(global_branch, local_branch)

    layer = []
    for i, m in enumerate(global_branch):
        if i > 7:
            break
        layer.append(m)

    model = nn.Sequential(*layer)

    torch.save(model, '/data1/wurundi/ML/baseline5_resnet50/model/best_features.pth.tar')

    local_branch = torch.load(LOCAL_BRANCH_DIR)['net'].module.children()
    layer = []
    for i, m in enumerate(local_branch):
        if i > 7:
            break
        layer.append(m)

    model = nn.Sequential(*layer)
    torch.save(model, '/data1/wurundi/ML/baseline5_resnet50_local/model/best_features.pth.tar')
    '''
    global_branch = torch.load(GLOBAL_BRANCH_DIR)['net'].module.state_dict()
    local_branch = torch.load(LOCAL_BRANCH_DIR)['net'].module.state_dict()
    net = fusenet(global_branch, local_branch)
    print(net)

if __name__ == '__main__':
    main()