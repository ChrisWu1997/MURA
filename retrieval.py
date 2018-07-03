from common import config
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import argparse
from dataset import get_dataloaders
import numpy as np
import os
from PIL import Image
import h5py


Trans = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_code(model, query_input):
    """
    :param model: the model to generate feature code
    :param query_input: PIL image for query
    :return: feature code for the image
    """
    inputs = Trans(query_input).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        feature = model(inputs, required_feature=True)
        code = F.avg_pool2d(feature, kernel_size=7)
        code = code.view(code.size(0), -1).cpu().numpy()
    return code


def generate_database(model, dataloader, save_dir):
    """
    :param model: model to generate feature code
    :param dataloader: dataloader to get dataset
    :param save_dir: the directory to save database.json
    :return: a code database:
                {'encounter name': {'file_path': filepath,
                                    'codes': codes}
                 ...
                }
    """
    data_codes = []
    data_filenames = []

    model.eval()
    pbar = tqdm(dataloader)
    for i, data in enumerate(pbar):
        inputs = data['image']
        img_filename = data['meta_data']['img_filename']
        inputs = inputs.to(config.device)

        with torch.no_grad():
            features = model(inputs, required_feature=True)
            codes = F.avg_pool2d(features, kernel_size=7)
            codes = codes.view(codes.size(0), -1).cpu().numpy()

        data_codes += [codes]
        data_filenames += img_filename

    data_codes = np.concatenate(data_codes)

    if save_dir is not None:
        f = h5py.File(os.path.join(save_dir, 'database.hdf5'), "w")
        f.create_dataset("codes", data=data_codes)

        asciiList = [n.encode("ascii", "ignore") for n in data_filenames]
        f.create_dataset('filenames', shape=(len(asciiList), 1), data=asciiList)

        print('database is generatred.')

    return


def retrieval(query_input, model, database, type):
    """
    :param query_input: input image for query
    :param model: the model to get feature (code)
    :param database: the code database to look up, of form hdf5
    :return: top 5 data entry that is similar to input
    """
    code = get_code(model, query_input)
    codes = np.array(database['codes'][:])
    filenames = np.array(database['filenames'][:])
    dist = []
    for _, c in enumerate(codes):
        dist.append(np.linalg.norm(code - c))

    dist = np.array(dist)
    rank = np.argsort(dist)
    cnt = 0
    outcome = []
    if type == 'ALL':
        outcome = rank[:5]
    else:
        for index in rank:
            if type in str(filenames[index]):
                outcome += [index]
                cnt += 1
                if cnt == 5:
                    break
    top5_filenames = filenames[outcome]
    return top5_filenames


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=config.data_dir, required=True, type=str,
                        help='parent directory of MURA-v1.0')
    parser.add_argument('-m', '--model_path', default='models/resnet50_b16.pth.tar', type=str, required=False,
                        help='filepath of the model')
    parser.add_argument('-d', '--database_path', type=str, required=False, help='filepath of the database',
                        default='database/database.hdf5')
    parser.add_argument('--save_dir', type=str, help='directory to write result database.json',
                        default='results/retrieval_result')
    parser.add_argument('--generate', action='store_true', help='generate database')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--img_path', type=str, required=True, help='filepath of query input')
    parser.add_argument('--img_type', default='ALL', type=str, required=False,
                        choices=['ELBOW', 'FINGER', 'FOREARM', 'HAND', 'HUMERUS', 'SHOULDER', 'WRIST', 'ALL'],
                        help='type of query input')
    args = parser.parse_args()

    net = torch.load(args.model_path)['net']

    if args.generate is True:
        dataloader = get_dataloaders('train', batch_size=args.batch_size, shuffle=False)

        generate_database(net, dataloader, args.save_dir)

    database = h5py.File(args.database_path, 'r')

    image = Image.open(args.img_path).convert('RGB')

    top5 = retrieval(image, net, database, args.img_type)
    print("The most similar five are:")
    i = 0
    for path in top5:
        i += 1
        print(path.item())
        path = os.path.join(args.data_dir, str(path.item())[2:-1])
        image = Image.open(path).convert('RGB')
        image.save(args.img_path[:-4] + str(i) + '.png')
