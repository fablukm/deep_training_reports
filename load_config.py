import json
import numpy as np
import os
from datetime import datetime
from pprint import pprint


def get_abbr_dict():
    abbr_dict = {'binary_crossentropy': 'bce',
                 'categorical_crossentropy': 'cce'}
    return abbr_dict


class Config(object):
    '''Load configuration for segmentation network.
    >> config = Config(fname)
    fname: string containing path to json file with parameters
    '''

    def __init__(self, filename='./config.json'):
        # insert metadata
        self.metadata = {}
        self.metadata['_filename'] = filename
        self.metadata['_abbr_dict'] = get_abbr_dict()
        self.metadata['_datetimeformat'] = "%d.%m.%Y_%H:%M:%S"
        self.metadata['root_dir'] = os.getcwd()
        self.metadata['timestamp'] = datetime.now().strftime(
            self.metadata['_datetimeformat'])

        # load config file
        self._load()

        # postprocess configuration
        self._postproc()
        return

    def _load(self):
        with open(self.metadata['_filename'], 'r') as f:
            conf_dict = json.load(f)
        for k, v in conf_dict.items():
            setattr(self, k, v)
        return

    def _postproc(self):
        abbr = self.metadata['_abbr_dict']

        # data split
        ratio_sum = np.abs(
            1-np.array([self.data[split]
                        for split in ['train_frac', 'test_frac', 'dev_frac']]).sum())
        assert ratio_sum < 1e-10, 'Invalid test-train-dev split'

        # filename for pickle
        pkl_fn = 'dataset_{}tr_{}te_{}dev.pickle'.format(self.data['train_frac'],
                                                         self.data['test_frac'],
                                                         self.data['dev_frac'])
        self.data['path_pickle'] = os.path.join(self.data['path_pickle'], pkl_fn)

        # check n_classes
        if not self.data['class_exclusive']:
            self.data['n_classes'] = 3
        else:
            self.data['n_classes'] = 4

        # image size to tuple
        self.data['image_size'] = tuple(self.data['image_size'])

        # generate filename
        self.model['filename_save'] = self.model['model_name'] + '_' + \
            str(self.data['train_frac']) + 'train_' + \
            abbr[self.training['loss']] + '_' + \
            str(self.training['n_epochs']) + 'ep.h5'

        # check if load file exists
        if self.model['is_pretrained']:
            model_file = os.path.join(
                self.model['model_folder'], self.model['filename_load'])
            assert os.path.exists(
                model_file), "Pretrained model file {} does not exist".format(model_file)
        return

    def print(self):
        pprint(self.__dict__)
        return self.__dict__


if __name__ == '__main__':
    print('LOADED CONFIG FILE TO OBJECT:\n')
    config = Config()
    pprint(config.__dict__)
