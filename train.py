from nn_model import NNModel
from load_config import Config
from dataset.dataset import GalileiSegmentationData
import models.models as models
from report import make_reports

import time

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed


def choose_model(config):
    if config.model['model_name'] == 'UnetSigmoid':
        model = models.unet_sigmoid(config)
    elif config.model['model_name'] == 'UnetSoftmax':
        model = models.unet_softmax(config)
    elif config.model['model_name'] == 'ResUnetSoftmax':
        model = models.resunet_softmax(config)
    else:
        raise NotImplementedError(
            'Model name {} not implemented.'.format(config.model['model_name']))
    return model


def train(config):
    model = choose_model(config)
    dataset = GalileiSegmentationData(config)
    net = NNModel(dataset, model, config)
    net.train()
    return net

if __name__=='__main__':
    json_folder = r'D:\NeuralNets\GalileiTopViewSegmenter'
    for configfile in ['./config_unet.json', './config_resunet.json']:
        config = Config(configfile)
        net = train(config)
        d = net.to_json()
    reports, uniquemodels = make_reports(json_dir=json_folder)

    # import tikzplotlib
    # import matplotlib.pyplot as plt
    #hist = net.training_history.history
    #[plt.plot(net.training_history.epoch, hist[key], label=key) for key in hist.keys()]
    #plt.legend()
    #tikzplotlib.save('test.tex', encoding='utf-8')
    #plt.close()
