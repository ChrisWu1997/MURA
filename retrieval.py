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
        transforms.Resize((256, 2256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.2056, 0.2056, 0.2056], [0.0313, 0.0313, 0.0313])
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_code(model, query_input):
    '''
    :param model: the model to generate feature code
    :param query_input: PIL image for query
    :return: feature code for the image
    '''
    inputs = Trans(query_input).unsqueeze(0)
    print('input size', inputs.size())
    model.eval()
    with torch.no_grad():
        feature = model(inputs, required_feature=True)
        code = F.avg_pool2d(feature, kernel_size=7)
        code = code.view(code.size(0), -1).cpu().numpy()
    return code


def generate_database(model, dataloader, save_dir):
    '''
    :param model: model to generate feature code
    :param dataloader: dataloader to get dataset
    :param save_dir: the directory to save database.json
    :return: a code database:
                {'encounter name': {'file_path': filepath,
                                    'codes': codes}
                 ...
                }
    '''
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

        data_codes.append(codes)
        data_filenames += img_filename

    data_codes = np.concatenate(data_codes)

    if save_dir is not None:
        f = h5py.File(os.path.join(save_dir, 'database.hdf5'), "w")
        f.create_dataset("codes", data=data_codes)

        asciiList = [n.encode("ascii", "ignore") for n in data_filenames]
        f.create_dataset('filenames', shape=(len(asciiList), 1), data=asciiList)

        print('database is generatred with shape:', f['codes'].shape)

    return


def retrieval(query_input, model, database):
    '''
    :param query_input: input image for query
    :param model: the model to get feature (code)
    :param database: the code database to look up, of form hdf5
    :return: top 5 data entry that is similar to input
    '''
    code = get_code(model, query_input)
    print('code shape', code.shape)
    codes = np.array(database['codes'][:])
    print('codes shape', codes.shape)
    filenames = np.array(database['filenames'][:])

    dist = []
    pbar = tqdm(codes)
    for _, c in enumerate(pbar):
        dist.append(np.linalg.norm(code - c))

    dist = np.array(dist)
    top5_index = np.argsort(dist)[:5]
    print('top5_index', top5_index)

    top5_filenames = filenames[top5_index]
    print('top5 filenames:', top5_filenames)
    print('top5 distance:', dist[top5_index])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_path', type=str, required=True, help='filepath of the model')
    parser.add_argument('-d', '--database_path', type=str, required=False, help='filepath of the database',
                        default='/data1/wurundi/ML/data/database.hdf5')
    parser.add_argument('--save_dir', type=str, help='directory to write result database.json',
                        default='/data1/wurundi/ML/data')
    parser.add_argument('--generate', action='store_true', help='generate database')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='mini-batch size')
    parser.add_argument('--img_path', type=str, required=False, help='filepath of query input')
    args = parser.parse_args()

    net = torch.load(args.model_path)['net']

    if args.generate is True:
        dataloader = get_dataloaders('train', batch_size=args.batch_size, shuffle=False)

        generate_database(net, dataloader, args.save_dir)

    database = h5py.File(args.database_path,'r')

    image = Image.open(args.img_path).convert('RGB')

    retrieval(image, net, database)

