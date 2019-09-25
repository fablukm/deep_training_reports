from nn_model import NNModel
from load_config import Config
from dataset.dataset import MNISTDataset
import models.models as models
from report import make_reports

import time

import os

def choose_model(config):
    if config['model']['name'] == 'ConvNet2layers':
        model = models.convnet2(config)
    elif config['model']['name'] == 'MLP2layers':
        model = models.mlp(config)
    else:
        raise NotImplementedError(
            'Model name {} not implemented.'.format(config['model']['name']))
    return model


def train(config):
    model = choose_model(config)
    dataset = MNISTDataset(config)
    net = NNModel(dataset, model, config)
    net.train()
    return net

if __name__=='__main__':
    json_folder = './training_logs'
    config_folder = './configs'
    doc = {'title': 'MNIST Training', 'author': 'Document Author'}
    for configfile in ['./configs/config_mlp2.json']:
        print('importing from {}...'.format(configfile))
        config = Config(configfile).config
        net = train(config)
        d = net.to_json()
    reports, uniquemodels = make_reports(json_dir=json_folder, doc=doc)
