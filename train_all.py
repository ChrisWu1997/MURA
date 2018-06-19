import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import numpy as np
import time
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter
import argparse
from model import network, DenseNet, densenet169
from common import config
from utils import TrainClock, save_args, AverageMeter
from dataset import get_dataloaders
from dataset import calc_data_weights

torch.backends.cudnn.benchmark = True
LOSS_WEIGHTS = calc_data_weights()

class Session:

    def __init__(self, config, net=None):
        self.log_dir = config.log_dir
        self.model_dir = config.model_dir
        self.net = net
        self.best_val_acc = 0.0
        self.tb_writer = SummaryWriter(log_dir=self.log_dir)
        self.clock = TrainClock()

    def save_checkpoint(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        tmp = {
            'net': self.net,
            'best_val_acc': self.best_val_acc,
            'clock': self.clock.make_checkpoint(),
        }
        torch.save(tmp, ckp_path)

    def load_checkpoint(self, ckp_path):
        checkpoint = torch.load(ckp_path)
        self.net = checkpoint['net']
        self.clock.restore_checkpoint(checkpoint['clock'])
        self.best_val_acc = checkpoint['best_val_acc']


def train_model(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter('epoch_loss')
    accs = AverageMeter('epoch_acc')

    # ensure model is in train mode
    model.train()
    pbar = tqdm(train_loader)
    for i, data in enumerate(pbar):
        inputs = data['image']
        labels = data['label']
        meta_datas = data['meta_data']
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        weights = [LOSS_WEIGHTS[labels[i]][meta_datas['study_type'][i]] for i in range(inputs.size(0))]
        weights = torch.Tensor(weights).view_as(labels).to(config.device)

        # pass this batch through our model and get y_pred
        outputs = model(inputs)
        preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
        #preds = torch.argmax(outputs, dim=1)

        # update loss metric
        loss = F.binary_cross_entropy(outputs, labels.float(), weights)
        #loss = criterion(outputs, labels)
        losses.update(loss.item(), inputs.size(0))

        corrects = torch.sum(preds.view_as(labels) == labels.float().data)
        acc = corrects.item() / inputs.size(0)
        accs.update(acc, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, i, len(train_loader)))
        pbar.set_postfix(
            loss=":{:.4f}".format(losses.avg),
            acc=":{:.4f}".format(accs.avg)
        )

    outspects = {
        'epoch_loss': losses.avg,
        'epoch_acc': accs.avg
    }
    return outspects


def valid_model(valid_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter('epoch_loss')
    accs = AverageMeter('epoch_acc')
    # ensure model is in train mode

    model.eval()
    pbar = tqdm(valid_loader)
    for i, data in enumerate(pbar):
        inputs = data['image']
        labels = data['label']
        meta_datas = data['meta_data']
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        weights = [LOSS_WEIGHTS[labels[i]][meta_datas['study_type'][i]] for i in range(inputs.size(0))]
        weights = torch.Tensor(weights).view_as(labels).to(config.device)

        with torch.no_grad():
            # pass this batch through our model and get y_pred
            outputs = model(inputs)
            preds = (outputs.data > 0.5).type(torch.cuda.FloatTensor)
            # preds = torch.argmax(outputs, dim=1)

            # update loss metric
            loss = F.binary_cross_entropy(outputs, labels.float(), weights)
            # loss = criterion(outputs, labels)
            losses.update(loss.item(), inputs.size(0))

            corrects = torch.sum(preds.view_as(labels) == labels.float().data)
            acc = corrects.item() / inputs.size(0)
            accs.update(acc, inputs.size(0))

        pbar.set_description("EPOCH[{}][{}/{}]".format(epoch, i, len(valid_loader)))
        pbar.set_postfix(
            loss=":{:.4f}".format(losses.avg),
            acc=":{:.4f}".format(accs.avg)
        )

    torch.cuda.empty_cache()

    outspects = {
        'epoch_loss': losses.avg,
        'epoch_acc': accs.avg
    }
    return outspects


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int, help='epoch number')
    parser.add_argument('-b', '--batch_size', default=240, type=int, help='mini-batch size')
    parser.add_argument('--lr', '--learning_rate', default=1e-2, type=float, help='initial learning rate')
    parser.add_argument('-c', '--continue', dest='continue_path', type=str, required=False)
    parser.add_argument('--exp_name', default=config.exp_name, type=str, required=False)
    parser.add_argument('--drop_rate', default=0, type=float, required=False)
    parser.add_argument('--only_fc', action='store_true', help='only train fc layers')
    parser.add_argument('--net', default='densenet169', type=str, required=False)
    args = parser.parse_args()
    print(args)

    config.exp_name = args.exp_name
    config.make_dir()
    save_args(args, config.log_dir)

    #net = network()
    if args.net == 'resnet50':
        net = models.resnet50(pretrained=True)
        net.fc = nn.Sequential(nn.Linear(2048,1), nn.Sigmoid())
    elif args.net == 'densenet121':
        net = models.densenet121(pretrained=True)
        net.classifier = nn.Sequential(nn.Linear(1024,1), nn.Sigmoid())
    else:
        net = densenet169(pretrained=True, drop_rate=args.drop_rate)

    net = torch.nn.DataParallel(net).cuda()
    sess = Session(config, net=net)

    train_loader = get_dataloaders('train', batch_size=args.batch_size, shuffle=True)

    valid_loader = get_dataloaders('valid', batch_size=args.batch_size, shuffle=False)

    if args.continue_path and os.path.exists(args.continue_path):
        sess.load_checkpoint(args.continue_path)

    clock = sess.clock
    tb_writer = sess.tb_writer

    #criterion = nn.CrossEntropyLoss().cuda()
    criterion = nn.BCELoss().cuda()

    if args.only_fc == True:
        optimizer = optim.Adam(sess.net.module.classifier.parameters(), args.lr)
    else:
        optimizer = optim.Adam(sess.net.parameters(), args.lr)

    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=0.1,  patience=10, verbose=True)

    for e in range(args.epochs):
        train_out = train_model(train_loader, sess.net,
                                criterion, optimizer, clock.epoch)
        valid_out = valid_model(valid_loader, sess.net,
                                criterion, optimizer, clock.epoch)

        tb_writer.add_scalars('loss',{'train': train_out['epoch_loss'],
                                      'valid': valid_out['epoch_loss']}, clock.epoch)

        tb_writer.add_scalars('acc',{'train': train_out['epoch_acc'],
                                      'valid': valid_out['epoch_acc']}, clock.epoch)

        tb_writer.add_scalar('learning_rate', optimizer.param_groups[-1]['lr'], clock.epoch)
        scheduler.step(valid_out['epoch_acc'])

        if valid_out['epoch_acc'] > sess.best_val_acc:
            sess.best_val_acc = valid_out['epoch_acc']
            sess.save_checkpoint('best_model.pth.tar')

        if clock.epoch % 10 == 0:
            sess.save_checkpoint('epoch{}.pth.tar'.format(clock.epoch))
        sess.save_checkpoint('latest.pth.tar')

        clock.tock()


if __name__ == '__main__':
    main()