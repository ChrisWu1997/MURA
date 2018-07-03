from common import config
from tqdm import tqdm
import torch
import argparse
from dataset import get_dataloaders
from utils import AUCMeter
import numpy as np
import os
from model import resnet50, densenet169, fusenet


model_list = ['models/resnet50_b16.pth.tar',
              'models/densenet169_b64.pth.tar',
              'models/densenet169_b32.pth.tar',
              'models/densenet169_b16.pth.tar',
              'models/fusenet_b16.pth.tar'
              ]


def get_scores(model, dataloader):
    """
    # using model to predict, based on dataloader
    :param model: the model to be tested on
    :param dataloader: suppose to be 'valid' or 'test' dataloader
    :return: the predict scores at study level
    """
    model.eval()

    st_corrects = {st: 0 for st in config.study_type}
    nr_stype = {st: 0 for st in config.study_type}
    study_out = {}  # study level output
    study_label = {}  # study level label
    auc = AUCMeter()

    # evaluate the model
    pbar = tqdm(dataloader)
    for k, data in enumerate(pbar):
        inputs = data['image']
        labels = data['label']
        encounter = data['meta_data']['encounter']
        study_type = data['meta_data']['study_type']
        inputs = inputs.to(config.device)
        labels = labels.to(config.device)

        outputs = model(inputs)

        for i in range(len(outputs)):
            if study_out.get(encounter[i], -1) == -1:
                study_out[encounter[i]] = [outputs[i].item()]
                study_label[encounter[i]] = labels[i].item()
                nr_stype[study_type[i]] += 1
            else:
                study_out[encounter[i]] += [outputs[i].item()]

    # study level prediction
    study_preds = {x: (np.mean(study_out[x]) > 0.5) == study_label[x] for x in study_out.keys()}

    for x in study_out.keys():
        st_corrects[x[:x.find('_')]] += study_preds[x]

    # acc for each study type
    avg_corrects = {st: st_corrects[st] / nr_stype[st] for st in config.study_type}

    total_corrects = 0
    total_samples = 0

    for st in config.study_type:
        print(st+' acc:{:.4f}'.format(avg_corrects[st]))
        total_corrects += st_corrects[st]
        total_samples += nr_stype[st]

    # acc for the whole dataset
    print('total acc:{:.4f}'.format(total_corrects / total_samples))
    
    # auc value
    final_scores = [np.mean(study_out[x]) for x in study_out.keys()]
    auc_output = np.array(final_scores)
    auc_target = np.array(list(study_label.values()))
    auc.add(auc_output, auc_target)

    auc_val, tpr, fpr = auc.value()
    print('AUC:{:.4f}'.format(auc_val))

    return study_out, study_label, nr_stype


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=config.data_dir, type=str,
                        help='parent directory of MURA-v1.0')
    parser.add_argument('--save_dir', default='results', type=str,
                        help='parent directory to write result')
    parser.add_argument('--phase', default='test', type=str, choices=['valid', 'test'])
    parser.add_argument('-b', '--batch_size', default=16, type=int, help='mini-batch size')
    args = parser.parse_args()

    dataloader = get_dataloaders(args.phase, batch_size=args.batch_size, shuffle=False, data_dir=args.data_dir)
    total_scores = None  # voting scores
    labels = None
    st_corrects = {st: 0 for st in config.study_type}
    nr_stype = {st: 0 for st in config.study_type}

    for j in range(len(model_list)):
        print('single model ' + str(j), model_list[j])

        if 'fuse' in model_list[j]:
            state_dict = torch.load(model_list[j])['state_dict']
            net = fusenet()
            net.load_state_dict(state_dict)
            net.set_fcweights()
            net = torch.nn.DataParallel(net).cuda()

        else:
            net = torch.load(model_list[j])['net']

        score, labels, nr_stype = get_scores(net, dataloader)
        if j == 0:
            total_scores = {x: score[x] for x in score.keys()}
        else:
            total_scores = {x: total_scores[x] + score[x] for x in total_scores.keys()}
        del net
        print('-' * 20)

    print('voting model:')
    f = open(os.path.join(args.save_dir, 'prediction_result.txt'), 'w')

    # study level prediction
    study_preds = {x: (np.mean(total_scores[x]) > 0.5) == labels[x] for x in total_scores.keys()}
    # study_preds = {x:study_preds[x] == study_label[x] for x in study_out.keys()}

    for x in total_scores.keys():
        st_corrects[x[:x.find('_')]] += study_preds[x]

    # acc for each study type
    avg_corrects = {st: st_corrects[st] / nr_stype[st] for st in config.study_type}

    total_corrects = 0
    total_samples = 0

    for st in config.study_type:
        print(st + ' acc:{:.4f}'.format(avg_corrects[st]))
        f.write(st+' acc:{:.4f}\n'.format(avg_corrects[st]))
        total_corrects += st_corrects[st]
        total_samples += nr_stype[st]

    # acc for the whole dataset
    print('total acc:{:.4f}'.format(total_corrects / total_samples))
    f.write('total acc:{:.4f}\n'.format(total_corrects / total_samples))

    # calculate voting classifier's auc
    auc = AUCMeter()
    final_scores = [np.mean(total_scores[x]) for x in total_scores.keys()]
    auc_output = np.array(final_scores)
    # auc_output = np.array(list(study_out.values()))a
    auc_target = np.array(list(labels.values()))
    auc.add(auc_output, auc_target)

    auc_val, tpr, fpr = auc.value()
    auc.draw_roc_curve(os.path.join(args.save_dir, 'ROC_curve.png'))

    print('VOTING AUC:{:.4f}'.format(auc_val))
    print('-' * 20)

    f.write('AUC:{:.4f}\n'.format(auc_val))
