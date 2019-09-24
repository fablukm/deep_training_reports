import os

from skimage import io, transform, filters, util

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class GalileiSegmentationData(object):
    '''
    ---- GalileiSegmenterDatset ----
    Dataset for Galilei TopView Segmentation data
    Implemented as enumerable object
    Augmentations are defined in method _get_augmentations

    INITIALISE:
        dataset = GalileiSegmenterDatset(config)
                  where config is a CONFIG object as loaded in load_config.py
                               and as defined in CONFIG.JSON (root folder)

    METHODS:
        dataset.__len__(): Returns length of dataset

    GET ITEM:
        img, mask = dataset[idx],
            where idx is the index

    USEFUL PUBLIC METHODS:
        self.visualise(do_plot=True, idx=-1)
            Visualise a sample of the data set.
            DO_PLOT: Boolean, set to true if sample needs to be plotted
            IDX: Int, index of the sample to be plotted.
                 Default: -1, corresponds to random selection of index.

    USEFUL PUBLIC ATTRIBUTES:
        df: Pandas DataFrame containing raw information about samples

    AUGMENTATIONS:
        Defined by self._get_augmentations
        returns aug_dict = {'functions': LIST OF FUNCTIONS,
                            'do_img': LIST OF INDICES TO BE APPLIED TO IMAGE,
                            'do_masks': LIST OF INDICES TO BE APPLIED TO MASK}
        Currently implemented:
            self.aug_dict['functions'] = [HORIZ_FLIP, VERT_FLIP, S&P_Noise]

    ENUMERATION:
        Given N raw data images with 3 masks and M augmentations.
        Length of dataset is M*N
        First N are original raw images
        (k+1)th N images are augmented by kth augmentation, k=1,...,M

        Access to N: self.df.__len__()
        Access to M: self.aug_dict['do_img'].__len__()

    (c) FMU for Ziemer Group, 2019
    '''

    def __init__(self, config):
        self.config = config
        self.path_parent = self.config.data['path']
        self.fn_json = self.config.data['data_json']
        self.image_size = config.data['image_size']

        self.augmentations = self._get_augmentations()
        self.df = self._get_df()
        self.path_imgs = os.path.join(config.data['path'], 'Image')
        self.path_masks = {'eyelids': os.path.join(config.data['path'], 'MasksEyelids'),
                           'limbus':  os.path.join(config.data['path'], 'MasksLimbus'),
                           'pupil':   os.path.join(config.data['path'], 'MasksPupil'),
                           'iris':    os.path.join(config.data['path'], 'MasksIris')}
        return

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError('Dataset index out of range')
        # load filename
        n_aug = self.augmentations['do_img'].__len__()
        n_data_raw = self.df.__len__()
        fn_img = self.df.iloc[idx % (n_data_raw-1)]['filename']
        path_img = os.path.join(self.path_imgs, fn_img)
        filestem = os.path.splitext(fn_img)[0]

        # load RGB channels of image and normalise
        img = io.imread(path_img)[:, :, 0:3]/255.

        # remove LED
        aux_sizes = [img.shape[1], img.shape[1], img.shape[0], img.shape[0]]
        x_start, x_end, y_start, y_end = [
            int(aux_sizes[k]*self.config.data['led_region'][k]) for k in range(4)]
        thr = self.config.data['led_threshold']
        led_mask = (img[y_start:y_end, x_start:x_end, :].min(axis=-1) > thr)
        img[y_start:y_end, x_start:x_end][led_mask >
                                          thr] = self.config.data['led_replace']*255.

        # load masks
        masks_keys = ['pupil', 'limbus', 'eyelids']
        masks_list = [io.imread(os.path.join(
            self.path_masks[key], filestem + '_' + key + '.png')) for key in masks_keys]

        # augment
        do_aug = format(idx//(n_data_raw-1), 'b').zfill(n_aug)

        for k in range(n_aug):
            # get index of augmentation
            idx_aug = self.augmentations['do_img'][k]

            # skip if current index does not have present augmentation
            if not int(do_aug[k]):
                continue

            # otherwise, perform augmentation on image...
            img = self.augmentations['functions'][idx_aug](img)

            # ... and on masks if required
            if idx_aug in self.augmentations['do_masks']:
                masks_list = [self.augmentations['functions']
                              [idx_aug](mask) for mask in masks_list]

        # resize and normalise
        masks_list = [transform.resize(mask, self.image_size)
                      for mask in masks_list]
        img = transform.resize(img, self.image_size)

        # stack masks to array
        masks = np.stack(masks_list, axis=2)

        if self.config.data['class_exclusive']:
            idx_pupil, idx_oz, idx_lid = 0, 1, 2
            for idx_mask in [idx_pupil, idx_oz]:
                masks[:, :, idx_mask] = (
                    1.-masks[:, :, idx_lid])*masks[:, :, idx_mask]
            masks[:, :, idx_oz] = (
                1.-masks[:, :, idx_pupil])*masks[:, :, idx_oz]

            sclera = np.expand_dims(1. - masks.sum(axis=2), axis=-1)
            masks = np.concatenate([masks, sclera], axis=-1)

        return img, masks

    def __len__(self):
        return self.df.__len__()*2**self.augmentations['do_img'].__len__()

    def _get_df(self):
        path_json = os.path.join(
            self.config.data['path'], self.config.data['data_json'])
        df = pd.read_json(path_json, orient='records')
        return df

    def _get_augmentations(self):
        conf = self.config.augmentations
        augmentations = [lambda x: x[:, ::-1],  # horizontal
                         lambda x: x[::-1, :],  # vertical
                         lambda x: transform.rotate(x, angle=conf['rotangle'], mode='edge'),
                         lambda x: util.random_noise(x, mode=conf['noisetype'])
                         ]
        aug_dict = {'functions': augmentations,
                    'do_img': list(range(augmentations.__len__())),
                    'do_masks': [0, 1, 2]}
        return aug_dict

    def split(self):
        tr, dev, test = get_dataset_split(self.__len__(), self.config)
        return {'training': tr,
                'dev': dev,
                'test': test}

    def visualise(self, do_plot=True, idx=-1):
        if idx == -1:
            idx = np.random.randint(low=0, high=self.__len__()-1)

        # compute which augmentations are done for output
        n_aug = self.augmentations['do_img'].__len__()
        do_aug = format(idx//(self.df.__len__()-1), 'b').zfill(n_aug)
        # load sample
        img, mask = self.__getitem__(idx)

        # initialise mask combination as zeros
        one_mask = np.zeros(img.shape)

        # colour mask combinations
        colours = [(245, 66, 66), (245, 224, 66),
                   (66, 150, 245), (69, 245, 66)]
        for idx_mask in range(4):
            for idx_col in range(3):
                one_mask[:, :, idx_col] += mask[:, :, idx_mask] * \
                    colours[idx_mask][idx_col]/255.

        orig_idx = idx % (self.df.__len__()-1)
        print('----------------------')
        print('Visualisation of a sample in Galilei TopView dataset')
        print('----')
        print('Index: {}'.format(idx))
        print('Corresponds to original index {} with augmentations {}'.format(
            orig_idx, ', '.join(do_aug)))
        print('----')

        if do_plot:
            print('Colour codes:    Red:    Pupil')
            print('                 Yellow: Iris')
            print('                 Blue:   Eyelids')
            print('                 Green:  Sclera')
            w = .8
            plt.figure()
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.subplot(2, 2, 3)
            plt.imshow(w*img+(1-w)*one_mask)
            plt.subplot(2, 2, 4)
            plt.imshow(one_mask)
            plt.show()
        print('----------------------')

        return img, one_mask


class BatchLoader(keras.utils.Sequence):
    def __init__(self, dataset, indices, batch_size=8, shuffle=False):

        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()
        return

    def __len__(self):
        return int(np.floor(len(self.indices)/float(self.batch_size)))

    def n_samples(self):
        return self.dataset.__len__()

    def __getitem__(self, idx):
        '''
        Load batch idx
        '''
        assert idx <= self.__len__(
        ), "Batch index out of bounds. Attempting to load item {} out of {}".format(idx, self.__len__())

        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        files_batch = np.arange(idx*self.batch_size, (idx+1)*self.batch_size)

        # todo: unnaive this part
        img, masks = [], []

        for idx in files_batch:
            _img, _masks = self.dataset[idx]
            img.append(_img)
            masks.append(_masks)

        return np.array(img), np.array(masks)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        return


def get_dataset_split(n_samples, config):
    '''
    idc_train, idc_dev, idc_test = get_dataset_split(n_samples, config):

    Split dataset of size n_samples into three parts. The configuration in
        the CONFIG object to be passed contains the ratios of the partition.

    INPUTS: n_samples: int>0. size of dataset to be split.
            config: CONFIG object as loaded by load_config in root folder, and
                    as specified by config.json in root folder.

    OUTPUTS: Three lists of integers containing the indices of the data
             samples corresponding to training set, validation set,
             and test set, respectively.

    CONFIG: config.data must contain keys train_frac, dev_frac, test_frac.
            Each of these values are a ratio between 0 and 1,
            corresponding to the part of the dataset attributed to
            training, dev, and test set, respectively.
            These values must sum up to 1 (enforced by config loader).
    '''
    idc = np.arange(n_samples)
    np.random.shuffle(idc)
    r_tr, r_val = config.data['train_frac'], config.data['dev_frac']
    idx_tr, idx_val = int(r_tr*n_samples), int((r_val+r_tr)*n_samples)
    idx_val = idx_val if idx_val-idx_tr > 0 else idx_tr+1

    idc_train = idc[:idx_tr]
    idc_val = idc[idx_tr:idx_val]
    idc_test = idc[idx_val:]

    assert len(idc_train)+len(idc_val) + \
        len(idc_test) == n_samples, "Dataset partition not covering dataset."
    assert min(len(idc_train), len(idc_val), len(idc_test)
               ) > 0, "Dataset partition contains empty set."
    return idc_train, idc_val, idc_test


def print_members():
    idx = 0
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            print('    '+str(idx)+': {}'.format(obj))
        idx += 1
    return


def test_galilei_set():
    sys.path.append('..')
    from load_config import Config
    conf = Config(filename='../config.json')
    print('--------------------')
    print('Testing class GalileiSegmentationData')
    print('--------------------')
    idx0 = 110
    ds = GalileiSegmentationData(conf)
    n_aug = ds.augmentations['do_img'].__len__()

    # +idx_aug*ds.df.__len__() for idx_aug in range(2**n_aug)]
    idc = [idx0+6*(189-1)]

    for idx in idc:
        img, mask = ds[idx]
        print('getting image {}...'.format(idx))
        plt.figure(idx)
        thr = 0.8
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    import sys
    import inspect
    import matplotlib.pyplot as plt
    import cProfile
    print('Dataset generator')
    print('Implemented:')
    print_members()

    print('--------------------')
    print('Importing configuration:')
    sys.path.append('..')
    from load_config import Config
    conf = Config(filename='../config.json')
    print('--------------------')
    print('Profiling Data loader:')
    dataset = GalileiSegmentationData(config=conf)
    idc_train, idc_dev, idc_test = get_dataset_split(dataset.__len__(), conf)
    train_loader = BatchLoader(dataset, idc_train)
    #cProfile.run('train_loader[5]')
    dataset.visualise()
