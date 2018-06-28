from common import config
from tqdm import tqdm
import torch
import argparse
from dataset import get_dataloaders
from utils import AUCMeter
import numpy as np
import os
from model import resnet50

def predict(model, dataloader):
    # using model to predict, based on dataloader
    model.eval()

    st_corrects = {st:0 for st in config.study_type}
    nr_stype = {st:0 for st in config.study_type}
    study_out = {}  # study level output
    study_label = {} # study level label
    auc = AUCMeter()

    # evaluate the model
    pbar = tqdm(dataloader)
    for i, data in enumerate(pbar):
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
    study_preds = {x:(np.mean(study_out[x]) > 0.5) == study_label[x] for x in study_out.keys()}
    #study_preds = {x:study_preds[x] == study_label[x] for x in study_out.keys()}

    for x in study_out.keys():
        st_corrects[x[:x.find('_')]] += study_preds[x]

    # acc for each study type
    print('st_corrects:', st_corrects)
    print('nr_stype', nr_stype)
    avg_corrects = {st:st_corrects[st] / nr_stype[st] for st in config.study_type}

    total_corrects = 0
    total_samples = 0
    f = open(os.path.join(args.save_dir, 'result.txt'), 'w')
    for st in config.study_type:
        print(st+' acc:{:.4f}'.format(avg_corrects[st]))
        f.write(st+' acc:{:.4f}\n'.format(avg_corrects[st]))
        total_corrects += st_corrects[st]
        total_samples += nr_stype[st]

    # acc for the whole dataset
    print('total acc:{:.4f}'.format(total_corrects / total_samples))
    f.write('total acc:{:.4f}\n'.format(total_corrects / total_samples))

    # auc value
    auc_output = np.array(list(study_out.values()))
    auc_target = np.array(list(study_label.values()))
    auc.add(auc_output, auc_target)

    auc_val, tpr, fpr = auc.value()
    auc.draw_roc_curve(os.path.join(args.save_dir, 'ROC_curve.png'))
    print('AUC:{:.4f}'.format(auc_val))
    f.write('AUC:{:.4f}\n'.format(auc_val))

    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='filepath of the model')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to write result')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='mini-batch size')
    args = parser.parse_args()

    net = torch.load(args.model_path)['net']

    dataloader = get_dataloaders('valid', batch_size=args.batch_size, shuffle=False)
    predict(net,dataloader)
