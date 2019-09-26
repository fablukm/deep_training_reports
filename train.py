from nn_wrapper import NeuralNetWrapper
from model_configs.load_config import Config
from dataset.dataset import *
import models.models as models
from report import make_reports
import json
import glob

import time

import os

def choose_model(model_config):
    if model_config['model']['name'] == 'ConvNet2layers':
        model = models.convnet2(model_config)
    elif model_config['model']['name'][:3]=='MLP':
        model = models.mlp(model_config)
    else:
        raise NotImplementedError(
            'Model name {} not implemented.'.format(model_config['model']['name']))
    return model


def train_model(model_config):
    model = choose_model(model_config)
    dataset = MNISTDataset(model_config)
    net = NNModel(dataset, model, model_config)
    net.train()
    return net

def main(report_config='./report_config.json'):
    '''
    TRAIN_ALL(report_config): Main file of the project.

    INPUT: report_config (string): String with filename (path)
           of the report configuration file.
           The report_config file is a .json containing information about
           what models to train.
           DEFAULT VALUE: './report_config.json'

    OUTPUT: None.
            This function write training logs and a final report as .tex
            and rendered .pdf in the path specified by the report_config file.

    '''
    # load json config
    with open(report_config, 'r') as handle:
        config = json.load(handle)

    # find model configs to train
    if len(config['models_to_train'])==0:
        search_pattern = os.path.join(config['model_config_folder'], '*.json')
        models_to_train = [fn for fn in glob.glob(search_pattern)]

    # train all models
    for model_config_file in models_to_train:
        model_config = Config(model_config_file).config
        model = choose_model(model_config)
        dataset = MNISTDataset(model_config)
        net = NeuralNetWrapper(dataset, model, model_config)
        net.train()
        net.to_json()

    # generate report
    _, _ = make_reports(train_log_dir=config['train_log_dir'],
                        doc=config['documenttitle'])
    return

if __name__=='__main__':
    main()
