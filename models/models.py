from keras import layers, models
import inspect
import sys


def convnet2(config):
    '''
    Implementation of a simple Convnet for MNIST classification
    INPUTS: config: Config object
            needs to contain fields:
                config['data']['image_size']: image size. MNIST: (28, 28)
                config['data']['filter_sizes']: At least two filter sizes (int)
                config['model']['dense_units']: Per hidden layer, number of neurons
                config['model']['dropout_rates']: Dropout rate per hidden layer
                config['model']['n_classes']: Number of output classes. MNIST: 10.
                config['data']['name']: Name
    '''
    # Load parameters from config
    image_size = config['data']['image_size']
    filter_sizes = config['model']['filter_sizes']
    dropout_rates = config['model']['dropout_rates']
    dense_units = config['model']['dense_units']
    n_classes = config['model']['n_classes']
    model_name = config['model']['name']

    # assert config to be valid
    assert len(filter_sizes) >= 2, "More filters need to be specified in model {}".format(
        model_name)
    assert len(dropout_rates) >= 2, "Invalid image size in model {}".format(
        model_name)
    if len(dropout_rates) == 1:
        # if only one dropout rate specified, use it for both dropout layers
        dropout_rates = [dropout_rates[0] for _ in range(2)]

    # define architecture (you can use Sequential instead)
    inputs = layers.Input((image_size[0], image_size[1], 1))
    c1 = layers.Conv2D(filter_sizes[0], kernel_size=(
        3, 3), activation='relu')(inputs)
    c2 = layers.Conv2D(filter_sizes[1], kernel_size=(
        3, 3), activation='relu')(c1)
    p1 = layers.MaxPool2D(pool_size=(2, 2))(c2)
    d1 = layers.Dropout(dropout_rates[0])(p1)
    f1 = layers.Flatten()(d1)
    de1 = layers.Dense(units=dense_units, activation='relu')(f1)
    d2 = layers.Dropout(dropout_rates[1])(de1)
    outputs = layers.Dense(units=n_classes, activation='softmax')(d2)

    # define model
    model = models.Model(inputs, outputs)
    model.name = model_name
    return model


def mlp(config):
    '''
    Implementation of a Multilayer perceptron for MNIST classification
    INPUTS: config: Config object
            needs to contain fields:
                config['data']['image_size']: image size. MNIST: (28, 28)
                config['model']['n_hidden_layers']: Number of hidden layers
                config['model']['dense_units']: Per hidden layer, number of neurons
                config['model']['dropout_rates']: Dropout rate per hidden layer
                config['model']['n_classes']: Number of output classes. MNIST: 10. nss.
                config['model']['name']: Name, e.g. 'MLP_{}'.format(n_hidden_layers)
    '''
    # Load parameters from config
    image_size = config['data']['image_size']
    n_hidden_layers = config['model']['n_hidden_layers']
    dense_units = config['model']['dense_units']
    dropout_rates = config['model']['dropout_rates']
    n_classes = config['model']['n_classes']
    model_name = config['model']['name']

    # assert config to be valid
    assert len(dense_units) >= n_hidden_layers, "Not enough layer sizes specified in model {}".format(model_name)
    assert len(dropout_rates) >= n_hidden_layers, "Not enough dropout rates specified in model {}".format(model_name)

    # define architecture (you can use Sequential instead)
    inputs = layers.Input((image_size[0], image_size[1], 1))
    x = layers.Flatten()(inputs)
    for k in range(n_hidden_layers):
        x = layers.Dense(units=dense_units[k])(x)
        x = layers.Dropout(dropout_rates[k])(x)
    outputs = layers.Dense(units=n_classes, activation='softmax')(x)

    # define model
    model = models.Model(inputs, outputs)
    model.name = model_name
    return model


def print_members():
    idx = 0
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            print('    '+str(idx)+': {}'.format(obj))
        idx += 1
    return


if __name__ == '__main__':
    print('--------------------------------')
    print('Models implementation models.py')
    print('--------------------------------')
    print('Members:')
    print_members()
