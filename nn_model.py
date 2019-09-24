import keras
from tensorflow.python.client import device_lib
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime
import cpuinfo

import load_config
#from dataset.dataset import BatchLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class NNModel(object):
    def __init__(self, dataset, model, config):
        self.dataset = dataset
        self.model = model
        self.config = config

    def train(self):
        # compile model
        self.model.compile(optimizer=self.config.training['optimizer'],
                           loss=self.config.training['loss'],
                           metrics=self.config.training['metrics'])

        # load weights
        if self.config.model['is_pretrained']:
            # if model pretrained: load weights
            self.load_pretrained()

        # define callbacks
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_categorical_accuracy', factor=0.2, patience=5)
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_categorical_accuracy', restore_best_weights=True, patience=5)
        tb = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False,
                                         embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

        # train model
        now = datetime.now()
        if self.config.data['use_loader']:
            # if we use a data loader (many augmentations): get dataloader
            # get data loaders
            self._get_dataloaders()
            self.training_history = \
                self.model.fit_generator(generator=self.training_loader,
                                         epochs=self.config.training['n_epochs'],
                                         validation_data=self.dev_loader,
                                         use_multiprocessing=self.config.training['use_multiprocessing'])
        else:
            # if we load the data set into RAM
            self._get_datasets()
            self.training_history = \
                self.model.fit(x=self.training_set['inputs'],
                               y=self.training_set['labels'],
                               epochs=self.config.training['n_epochs'],
                               validation_data=(
                               self.test_set['inputs'], self.test_set['labels']),
                               batch_size=self.config.training['batch_size'],
                               shuffle=self.config.training['shuffle'],
                               callbacks=[reduce_lr, early_stopping],
                               verbose=1)
        # train model
        self.trainingtime = datetime.now() - now
        self.training_start = now.strftime(
            self.config.report['datetimeformat'])

        # save weights
        if self.config.model['is_saved']:
            self.save_weights()

        return

    def load_pretrained(self):
        weights_path = os.path.join(self.config.model['model_folder'],
                                    self.config.model['filename_load'])
        try:
            self.model.load_weights(weights_path)
            print('LOADED weights from {}'.format(weights_path))
        except:
            print('Could not load weights from {}'.format(weights_path))
        return

    def save_weights(self):
        weights_path = os.path.join(self.config.model['model_folder'],
                                    self.config.model['filename_save'])
        try:
            self.model.save_weights(weights_path)
            print('SAVED weights to {}'.format(weights_path))
        except:
            print('Could not save weights to {}'.format(weights_path))
        return

    def _get_datasets(self):
        # if no dataloader is used:
        if self.config.data['use_pickle']:
            # if we use a pre-saved dataset split:
            with open(self.config.data['path_pickle'], 'rb') as handle:
                print('Loading pickled Dataset from {}...'.format(
                    self.config.data['path_pickle']))
                samples = pickle.load(handle)
                print('    Dataset imported.')
        else:
            print('Loading Dataset anew from {}...'.format(
                self.config.data['path']))
            # if we don't load the pickled data dict, generate it (can take long time)
            idx_split = self.dataset.split()
            samples = {dstype: [self.dataset[idx] for idx in idx_split[dstype]]
                       for dstype in ['training', 'dev', 'test']}
            with open(self.config.data['path_pickle'], 'wb') as handle:
                pickle.dump(samples, handle)
            print('SAVED Dataset to {}...'.format(
                self.config.data['path_pickle']))
        # now extract the training set to a form compatible with model.fit
        inputs_tr, masks_tr = split_list_of_tuples(samples['training'])
        inputs_dev, masks_dev = split_list_of_tuples(samples['dev'])
        inputs_te, masks_te = split_list_of_tuples(samples['test'])
        self.training_set = {'inputs': inputs_tr, 'labels': masks_tr}
        self.dev_set = {'inputs': inputs_dev, 'labels': masks_dev}
        self.test_set = {'inputs': inputs_te, 'labels': masks_te}
        return

    def _get_dataloaders(self):
        self.data_partition = self.dataset.split()
        self.training_loader = \
            BatchLoader(dataset=self.dataset,
                        indices=self.data_partition['training'],
                        batch_size=self.config.training['batch_size'],
                        shuffle=self.config.training['shuffle'])

        self.dev_loader = \
            BatchLoader(dataset=self.dataset,
                        indices=self.data_partition['dev'],
                        batch_size=self.config.training['batch_size'],
                        shuffle=self.config.training['shuffle'])

        self.test_loader = \
            BatchLoader(dataset=self.dataset,
                        indices=self.data_partition['test'],
                        batch_size=self.config.training['batch_size'],
                        shuffle=self.config.training['shuffle'])
        return

    def plot_predictions(self, idc=None, n_samples=4, show=True):
        x_test = self.test_set['inputs']
        y_test = self.test_set['labels']
        if not idc:
            idc = np.random.randint(
                low=0, high=x_test.shape[0]-1, size=n_samples)

        plt.figure()
        n_onerow = 3
        w_mask = .2
        idx_row = 0
        for idx_sample in idc:
            img, gt = x_test[idx_sample, :], y_test[idx_sample, :]
            scores = self.model.evaluate(np.expand_dims(img, axis=0),
                                         np.expand_dims(gt, axis=0),
                                         verbose=0)
            pred = self.model.predict(np.expand_dims(img, axis=0),
                                      verbose=0).squeeze()

            gt_overlay = w_mask*gt[:, :, :3] + (1-w_mask)*img
            pred_overlay = w_mask*pred[:, :, :3] + (1-w_mask)*img

            gt_mask = np.argmax(gt, axis=-1)
            pred_mask = np.argmax(pred, axis=-1)

            summary = 'Loss: {}\nCat. accuracy: {}'.format(
                scores[0], scores[1])

            idx0 = idx_row*n_onerow + 1
            plt.subplot(n_samples, n_onerow, idx0)
            plt.imshow(img)
            plt.title(u'Sample #{}'.format(idx_sample))

            plt.subplot(n_samples, n_onerow, idx0+1)
            plt.imshow(gt_mask)
            plt.title('Ground Truth')

            plt.subplot(n_samples, n_onerow, idx0+2)
            plt.imshow(pred_mask)
            plt.title(u'Predicition: {}'.format(summary))
            idx_row += 1
        if show:
            plt.show()
        return

    def _evaluate_model_str(self):
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
        training_dict = {'epochs': self.config.training['n_epochs'],
                         'optimizer': self.config.training['optimizer'],
                         'loss': self.config.training['loss'],
                         'metrics': self.config.training['metrics'],
                         'batch_size': self.config.training['batch_size'],
                         'shuffle': self.config.training['shuffle'],
                         'history': hist_str,
                         'test_accuracies': self._evaluate_model_str()}
        # data related to the data set
        data_dict = {'image_size': self.config.data['image_size'],
                     'led_region': self.config.data['led_region'],
                     'led_threshold': self.config.data['led_threshold'],
                     'led_replace': self.config.data['led_replace'],
                     'class_exclusive': self.config.data['class_exclusive'],
                     'path_pickle': self.config.data['path_pickle'],
                     'aug_noisetype': self.config.augmentations['noisetype'],
                     'aug_rotangle': self.config.augmentations['rotangle'],
                     'samples': {'train': {'ratio': self.config.data['train_frac'],
                                           'n_samples': self.training_set['inputs'].shape[0]},
                                 'test':  {'ratio': self.config.data['test_frac'],
                                           'n_samples': self.test_set['inputs'].shape[0]},
                                 'dev':  {'ratio': self.config.data['dev_frac'],
                                          'n_samples': self.dev_set['inputs'].shape[0]},
                                 }
                     }

        # data related to the model architecture
        model_dict = {'name': self.model.name,
                      'n_params': self.model.count_params(),
                      'is_saved': self.config.model['is_saved'],
                      'is_pretrained': self.config.model['is_pretrained'],
                      'layers': []}

        if self.config.model['is_saved']:
            model_dict['save_folder'] = self.config.model['model_folder']
            model_dict['weights_filename'] = self.config.model['filename_save']

        if self.config.model['is_pretrained']:
            model_dict['pretrained_file'] = self.config.model['filename_load']

        for layer in self.model.layers:
            l_dict = {'name': layer.name,
                      'layertype': str(layer).split(' ')[0].split('.')[-1],
                      'n_params': layer.count_params(),
                      'output_shape': layer.output_shape[1:],
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

            elif l_dict['layertype'] == 'Add':
                pass

            elif l_dict['layertype'] == 'InputLayer':
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
                'timeformat_json': self.config.report['datetimeformat'],
                'timeformat_pdf': self.config.report['datetimeformat_report'],
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
        filepath = os.path.join(self.config.report['export_folder'],
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
    """
    Takes a timedelta object and formats it for humans.
    Usage:
        # 149 day(s) 8 hr(s) 36 min 19 sec
        print human_delta(datetime(2014, 3, 30) - datetime.now())
    Example Results:
        23 sec
        12 min 45 sec
        1 hr(s) 11 min 2 sec
        3 day(s) 13 hr(s) 56 min 34 sec
    :param tdelta: The timedelta object.
    :return: The human formatted timedelta
    """
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
    print('Training procedure implemented in class NNModel')
