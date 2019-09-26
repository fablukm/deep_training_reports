from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt


class MNISTDataset(object):
    '''
    Dummy Dataset to illustrate how the data integrates into the framework.
    INPUT: config: This is in general necessary to specify train-dev-test split,
                        augmentations, image size, and such.
                   Here, we merely load MNIST from keras, so no config needed.

    MANDATORY ATTRIBUTES:
        This class *must* contain attributes
            self.training_set, self.test_set, self.dev_set
        Each of them is a dictionary with keys 'inputs' and 'labels', with
            values being the inputs and labels, respectively.

    METHODS:
        visualise: Visualise random or given items from the dataset.
        __getitem__: This class is implemented as enumerable.
                     __getitem__(idx) defines what will be the idx-th element
        __len__: Defines length of dataset

    PRIVATE METHODS:
        _get_augmentations: if custom datasets are loaded, here is where
                            you define augmentations.
        _train_test_split: if custom datasets are loaded, here is where you
                           define which samples belong to
                           train/test/dev set, respectively.

    NOTE ABOUT THE IMPLEMENTATION OF AUGMENTATIONS:
        In my implementation, I load a number n of given images.
        In _get_augmentations, I define a list of augmentations,
            say n_aug augmentations.
        The dataset implementation pretends that the dataset and all
            combinations of augmentations are concatenated, ie 'stacked' on top
            of each other.
        Thus, the length of my dataset is n * 2^n_aug.
        Given an index as input to __getitem__, I convert the index to
            a binary number of n_aug digits, each digit determines whether the
            respective augmentation is performed or not.

        This loads up your RAM, but is considerably faster than a dataloader
        since I am working with a lot of RAM and small datasets.
    '''

    def __init__(self, config):
        self.config = config

        img_size = config['data']['image_size']

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train.reshape(
            [-1, img_size[0], img_size[1], 1]).astype('float32')
        x_test = x_test.reshape(
            [-1, img_size[0], img_size[1], 1]).astype('float32')
        x_train /= 255
        x_test /= 255
        y_train = to_categorical(y_train, config['model']['n_classes'])
        y_test = to_categorical(y_test, config['model']['n_classes'])

        self.training_set = {'inputs': x_train, 'labels': y_train}
        self.test_set = {'inputs': x_test, 'labels': y_test}
        self.dev_set = {'inputs': np.array([]), 'labels': np.array([])}

    def _get_augmentations(self):
        # augmentations can be set here
        pass

    def _train_test_split(self):
        # train test split (with random index selection) can be set here
        pass

    def __len__(self):
        return self.training_set['labels'].shape[0] + \
            self.test_set['labels'].shape[0]

    def __getitem__(self, idx):
        '''
        Dataset class can be implemented as enumerable.
        In MNIST, Keras loads and splits already, so this
        method just backtracks this process.
        '''
        if idx >= self.__len__():
            raise IndexError('Dataset index out of range')

        n_train = self.training_set['labels'].shape[0]
        if idx < n_train:
            sample = self.training_set['inputs'][idx, :]
            label = self.training_set['labels'][idx]
        else:
            acutal_idx = idx % n_train
            sample = self.test_set['inputs'][acutal_idx, :]
            label = self.test_set['labels'][acutal_idx]

        return sample, label

    def visualise(self, idc='random', do_plot=True):
        '''
        Dataset.visualise(idc, do_plot) visualises four samples.
        INPUTS:
            idc: if a list of four numbers, these numbers will be shown
                 if 'random', four random examples will be show.
            do_plot: decides whether a plt.show() is executed in the end.
                     default: true
        OUTPUTS: None.
        '''
        n_samples = 4  # hardcoded for subplot simplicity and minor importance.
        plt.figure()
        for i_sample in range(n_samples):
            plt.subplot(2, 2, i_sample+1)

            # select index
            if idc == 'random':
                idx = np.random.randint(low=0, high=self.__len__()-1)
            elif len(idc) == 4:
                idx = idc[i_sample]
            else:
                raise ValueError(
                    'Input \'idc\' to Dataset.visualise() invalid')

            n_train = self.training_set['labels'].shape[0]
            idx_plt = '{} (Train)'.format(
                idx) if idx < n_train else '{} (Test)'.format(idx % n_train)

            # read and plot data
            img, label = self.__getitem__(idx)
            plt.imshow(img)
            plt.axis('off')
            plt.title('Index: {}, Label: {}'.format(idx_plt, label))
        if do_plot:
            plt.show()
        return


if __name__ == '__main__':
    print('DATASET class:')
    print('    Shows implementation of dataset fitting into framework.')
    print('Generate plots for example: MNIST.')
    dataset = MNISTDataset(config='')
    dataset.visualise()
