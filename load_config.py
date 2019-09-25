import json
import numpy as np
import os
from datetime import datetime
from pprint import pprint


def get_abbr_dict():
    # Define abbreviations in case the loss needs to be inside the filename
    abbr_dict = {'binary_crossentropy': 'bce',
                 'categorical_crossentropy': 'cce'}
    return abbr_dict


class Config(object):
    '''Load configuration for segmentation network.
    >> config = Config(fname)
    fname: string containing path to json file with parameters
    '''

    def __init__(self, filename):

        self._filename = filename

        # load config file
        with open(filename, 'r') as f:
            self.config = json.load(f)

        # postprocess configuration
        self._postproc()
        return

    def _load(self):
        # load json into a dict
        return

    def _postproc(self):
        # if necessary, some entries of the json file can be modified

        # generate filename
        modelname = self.config['model']['name']
        n_ep = self.config['training']['n_epochs']
        self.config['model']['filename_save'] = '{}_{}ep_MNIST.h5'.format(
            modelname, n_ep)

        # insert metadata
        self.config['metadata'] = {}
        self.config['metadata']['_filename'] = self._filename
        self.config['metadata']['_abbr_dict'] = get_abbr_dict()
        self.config['metadata']['_datetimeformat'] = "%d.%m.%Y_%H:%M:%S"
        self.config['metadata']['root_dir'] = os.getcwd()
        self.config['metadata']['timestamp'] = datetime.now().strftime(
            self.config['metadata']['_datetimeformat'])
        return

    def print(self):
        # print the dictonary
        pprint(self.config)
        return


if __name__ == '__main__':
    print('LOADED CONFIG FILE TO OBJECT:\n')
    config = Config('./configs/config_convnet2.json')
    config.print()
