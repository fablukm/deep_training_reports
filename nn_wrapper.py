import keras
from tensorflow.python.client import device_lib
from keras.callbacks import ReduceLROnPlateau
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
import cpuinfo

def _optimizer_loader(keyword):
    dict = {'sgd': keras.optimizers.SGD,
            'rmsprop': keras.optimizers.RMSprop,
            'adagrad': keras.optimizers.Adagrad,
            'adadelta': keras.optimizers.Adadelta,
            'adam': keras.optimizers.Adam,
            'adamax': keras.optimizers.Adamax,
            'nadam': keras.optimizers.Nadam}
    return dict[keyword]

class NeuralNetWrapper(object):
    def __init__(self, dataset, model, config):
        self.dataset = dataset
        self.model = model
        self.config = config

    def train(self):
        # get optimizer
        try:
            optimizer = _optimizer_loader(self.config['training']['optimizer'])(**self.config['training']['optim_config'])
        except KeyError:
            raise NotImplementedError('Optimizer {} has not yet been implemented in nn_model/_optimizer_loader'.format(self.config['training']['optimizer']))
        # compile model
        self.model.compile(optimizer=self.config['training']['optimizer'],
                           loss=self.config['training']['loss'],
                           metrics=self.config['training']['metrics'])

        # Load the dataset
        self._get_datasets()

        # TRAINING
        #callbacks
        #TODO: make configurable
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=2)

        now = datetime.now()
        self.training_history = \
            self.model.fit(x=self.training_set['inputs'],
                           y=self.training_set['labels'],
                           epochs=self.config['training']['n_epochs'],
                           validation_data=(self.test_set['inputs'], self.test_set['labels']),
                           batch_size=self.config['training']['batch_size'],
                           shuffle=self.config['training']['shuffle'],
                           callbacks=[reduce_lr],
                           verbose=1)
        # train model
        self.trainingtime = datetime.now() - now
        self.training_start = now.strftime(
            self.config['report']['datetimeformat'])

        # save weights if necessary
        if self.config['model']['is_saved']:
            self.save_weights()

        return

    def load_pretrained(self):
        # if pretrained model is used
        pass

    def save_weights(self):
        weights_path = os.path.join(self.config['model']['model_folder'],
                                    self.config['model']['filename_save'])
        try:
            self.model.save_weights(weights_path)
            print('SAVED weights to {}'.format(weights_path))
        except:
            print('Could not save weights to {}'.format(weights_path))
        return

    def _get_datasets(self):
        # Here, we use the MNIST model shipped with keras
        # For custom datasets, this function may actually be useful
        self.training_set = self.dataset.training_set
        self.test_set = self.dataset.test_set
        self.dev_set = self.dataset.dev_set
        return

    def _get_dataloaders(self):
        # in case you want to use a dataloader instead
        pass

    def plot_predictions(self, idc='random', do_plot=True):
        '''
        net.plot_predictions(idc, do_plot) visualises four predictions.
        INPUTS:
            idc: if a list of four numbers, these numbers will be shown
                 if 'random', four random examples will be show.
            do_plot: decides whether a plt.show() is executed in the end.
                     default: true
        OUTPUTS: None.
        '''
        n_samples = 4  # hardcoded for subplot simplicity and minor importance.
        x_test, y_test = self.test_set['inputs'], self.test_set['labels']
        if idc=='random':
            idc = np.random.randint(low=0, high=x_test.shape[0]-1, size=n_samples)

        plt.figure()
        for i_sample in range(n_samples):
            idx = idc[i_sample]
            plt.subplot(2, 2, i_sample+1)

            img, label = x_test[idx, :], y_test[idx, :]

            pred = np.argmax(self.model.predict(np.expand_dims(img, axis=0),
                                      verbose=0).squeeze())

            # plot now
            plt.imshow(np.concatenate(3*[img], axis=-1))
            plt.axis('off')
            label = np.argmax(label)
            plt.title('Sample: {}: Label={}, Prediction={}'.format(idx, i_sample, label))

        if do_plot:
            plt.show()
        return

    def _evaluate_model_str(self):
        # this should not be done for large datasets. mnist itself is already too large
        accs = [str(self.model.evaluate(np.expand_dims(self.test_set['inputs'][idx, :], axis=0),
                                        np.expand_dims(
                                            self.test_set['labels'][idx, :], axis=0),
                                        verbose=0)[1])
                for idx in range(self.test_set['inputs'].shape[0])]
        return accs

    def to_json(self):
        # history dict to string
        hist_str = {key: [str(entry) for entry in self.training_history.history[key]]
                    for key in self.training_history.history.keys()}

        # data related to model training
        training_dict = {'epochs': self.config['training']['n_epochs'],
                         'optimizer': self.config['training']['optimizer'],
                         'optim_config_spec': self.config['training']['optim_config'],
                         'optim_config': self.model.optimizer.get_config(),
                         'loss': self.config['training']['loss'],
                         'metrics': self.config['training']['metrics'],
                         'batch_size': self.config['training']['batch_size'],
                         'shuffle': self.config['training']['shuffle'],
                         'history': hist_str,
                         #'test_accuracies': self._evaluate_model_str()
                         }

        # data related to the data set
        data_dict = {'name': self.config['data']['name'],
                     'image_size': self.config['data']['image_size'],
                     'samples': {'train': {'n_samples': self.training_set['inputs'].shape[0]},
                                 'test':  {'n_samples': self.test_set['inputs'].shape[0]},
                                 'dev':  {'n_samples': self.dev_set['inputs'].shape[0]},
                                 }
                     }

        # data related to the model architecture
        model_dict = {'name': self.model.name,
                      'n_params': self.model.count_params(),
                      'is_saved': self.config['model']['is_saved'],
                      'layers': []}

        if self.config['model']['is_saved']:
            model_dict['save_folder'] = self.config['model']['model_folder']
            model_dict['weights_filename'] = self.config['model']['filename_save']

        for layer in self.model.layers:
            l_dict = {'name': layer.name,
                      'layertype': str(layer).split(' ')[0].split('.')[-1],
                      'n_params': layer.count_params(),
                      'output_shape': str(layer.output_shape[1:]),
                      'inbound_layers': layer._inbound_nodes[0].get_config()['inbound_layers']
                      }

            layerconf = layer.get_config()
            if l_dict['layertype'] == 'Conv2D':
                l_dict['activation'] = layerconf['activation']
                l_dict['kernel_size'] = layerconf['kernel_size']
                l_dict['dilation_rate'] = layerconf['dilation_rate']
                l_dict['strides'] = layerconf['strides']
                l_dict['padding'] = layerconf['padding']
                l_dict['n_filters'] = layerconf['filters']

            elif l_dict['layertype'] == 'Concatenate':
                l_dict['concat_axis'] = layerconf['axis']

            elif l_dict['layertype'] == 'UpSampling2D':
                l_dict['scale_factors'] = layerconf['size']
                l_dict['interpolation'] = layerconf['interpolation']

            elif l_dict['layertype'] == 'MaxPooling2D':
                l_dict['pool_size'] = layerconf['pool_size']
                l_dict['padding'] = layerconf['padding']
                l_dict['strides'] = layerconf['strides']

            elif l_dict['layertype'] == 'BatchNormalization':
                l_dict['momentum'] = layerconf['momentum']
                l_dict['center'] = "Yes" if layerconf['center'] else "No"
                l_dict['scale'] = "Yes" if layerconf['scale'] else "No"
                l_dict['epsilon'] = layerconf['epsilon']
                l_dict['axis'] = layerconf['axis']

            elif l_dict['layertype'] == 'Activation':
                l_dict['activation'] = layerconf['activation']

            elif l_dict['layertype'] == 'Dropout':
                l_dict['activation'] = layerconf['rate']

            elif l_dict['layertype'] == 'Dense':
                l_dict['n_nodes'] = layerconf['units']
                l_dict['activation'] = layerconf['activation']

            elif l_dict['layertype'] in ['Flatten', 'Add', 'InputLayer']:
                pass

            else:
                print('SKIPPING: Export of layer type {} not implemented.'.format(
                    l_dict['layertype']))
            model_dict['layers'].append(l_dict)

        # save metadata

        meta = {'starttime': self.training_start,
                'training_time': format_timedelta(self.trainingtime),
                'training_device': get_device_info(),
                'keras_version': keras.__version__,
                'keras_backend': keras.backend.backend(),
                'tensorflow_version':  keras.backend.tensorflow_backend.tf.__version__,
                'timeformat_json': self.config['report']['datetimeformat'],
                'timeformat_pdf': self.config['report']['datetimeformat_report'],
                }

        # generate dict
        export_dict = {'model': model_dict,
                       'data': data_dict,
                       'training': training_dict,
                       'metadata': meta}

        # debug
        print('dictionary generated.')
        self.as_dict = export_dict

        # write to json
        filename = filename = 'training_{}_{}.json'.format(
            self.model.name.lower(), self.training_start)
        filepath = os.path.join(self.config['report']['export_folder'],
                                filename)
        try:
            with open(filepath, 'w') as handle:
                json.dump(export_dict, handle)
        except:
            print('ERROR: Cannot export to file {}'.format(filepath))

        return export_dict


def split_list_of_tuples(lot):
    # list of tuples to np arrays
    x_list, y_list = map(np.array, zip(*lot))
    return x_list, y_list


def format_timedelta(tdelta):
    d = dict(days=tdelta.days)
    d['hrs'], rem = divmod(tdelta.seconds, 3600)
    d['min'], d['sec'] = divmod(rem, 60)

    if d['min'] is 0:
        fmt = '{sec} sec'
    elif d['hrs'] is 0:
        fmt = '{min} min {sec} sec'
    elif d['days'] is 0:
        fmt = '{hrs} hr(s) {min} min {sec} sec'
    else:
        fmt = '{days} day(s) {hrs} hr(s) {min} min {sec} sec'

    return fmt.format(**d)


def get_device_info():
    devices = device_lib.list_local_devices()
    gpu_used = 'GPU' in str(devices)
    if gpu_used:
        device = str(devices[-1]).split('name: ')[-1].split(', ')[0]
    else:
        device = 'CPU'
    out_dict = {'gpu_used': gpu_used, 'device': device}

    cpu_info = cpuinfo.get_cpu_info()
    out_dict['python_version'] = cpu_info['python_version']
    out_dict['cpu'] = {'arch': cpu_info['arch'],
                       'brand': cpu_info['brand']}
    return out_dict


if __name__ == '__main__':
    print('MODULE nn_wrapper')
