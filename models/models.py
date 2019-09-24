from models.layers import *
import keras
import sys
import inspect
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed

def unet_softmax(config):
    '''
    unet(config)

    Implementation of Unet with four downsampling and upsampling blocks,
    respectively. Bottleneck and
    '''
    filter_sizes = config.model['filter_sizes']
    image_size = config.data['image_size']
    n_classes = config.data['n_classes']
    inputs = keras.layers.Input((image_size[0], image_size[1], 3))

    p0 = inputs
    c1, p1 = block_downsampling(p0, filter_sizes[0])  # 128 -> 64
    c2, p2 = block_downsampling(p1, filter_sizes[1])  # 64 -> 32
    c3, p3 = block_downsampling(p2, filter_sizes[2])  # 32 -> 16
    c4, p4 = block_downsampling(p3, filter_sizes[3])  # 16 -> 8

    bn = bottleneck(p4, filter_sizes[4])

    u1 = block_upsampling(bn, c4, filter_sizes[3])  # 8 -> 16
    u2 = block_upsampling(u1, c3, filter_sizes[2])  # 16 -> 32
    u3 = block_upsampling(u2, c2, filter_sizes[1])  # 32 -> 64
    u4 = block_upsampling(u3, c1, filter_sizes[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(
        n_classes, (1, 1), padding="same", activation="softmax")(u4)
    model = keras.models.Model(inputs, outputs)
    model.name='Unet'
    return model

def resunet_softmax(config):
    '''
    resunet(config)

    Implementation of ResUnet Encoder-Decoder architecture.
    '''
    filter_sizes = config.model['filter_sizes']
    image_size = config.data['image_size']
    n_classes = config.data['n_classes']
    inputs = keras.layers.Input((image_size[0], image_size[1], 3))

    ## Encoder
    e0 = inputs
    e1 = stem_resnet(e0, filter_sizes[0])
    e2 = residual_block(e1, filter_sizes[1], strides=2)
    e3 = residual_block(e2, filter_sizes[2], strides=2)
    e4 = residual_block(e3, filter_sizes[3], strides=2)
    e5 = residual_block(e4, filter_sizes[4], strides=2)

    ## Bridge
    b0 = conv_block(e5, filter_sizes[4], strides=1)
    b1 = conv_block(b0, filter_sizes[4], strides=1)

    ## Decoder
    u1 = upsample_concat_block(b1, e4)
    d1 = residual_block(u1, filter_sizes[4])

    u2 = upsample_concat_block(d1, e3)
    d2 = residual_block(u2, filter_sizes[3])

    u3 = upsample_concat_block(d2, e2)
    d3 = residual_block(u3, filter_sizes[2])

    u4 = upsample_concat_block(d3, e1)
    d4 = residual_block(u4, filter_sizes[1])

    outputs = keras.layers.Conv2D(n_classes, (1, 1), padding="same", activation="softmax")(d4)
    model = keras.models.Model(inputs, outputs)
    model.name="ResUnet"
    return model


def unet_sigmoid(config):
    '''
    unet(config)

    Implementation of Unet with four downsampling and upsampling blocks,
    respectively. Bottleneck and
    '''
    filter_sizes = config.model['filter_sizes']
    image_size = config.data['image_size']
    n_classes = config.data['n_classes']
    inputs = keras.layers.Input((image_size[0], image_size[1], 3))

    p0 = inputs
    c1, p1 = block_downsampling(p0, filter_sizes[0])  # 128 -> 64
    c2, p2 = block_downsampling(p1, filter_sizes[1])  # 64 -> 32
    c3, p3 = block_downsampling(p2, filter_sizes[2])  # 32 -> 16
    c4, p4 = block_downsampling(p3, filter_sizes[3])  # 16 -> 8

    bn = bottleneck(p4, filter_sizes[4])

    u1 = block_upsampling(bn, c4, filter_sizes[3])  # 8 -> 16
    u2 = block_upsampling(u1, c3, filter_sizes[2])  # 16 -> 32
    u3 = block_upsampling(u2, c2, filter_sizes[1])  # 32 -> 64
    u4 = block_upsampling(u3, c1, filter_sizes[0])  # 64 -> 128

    outputs = keras.layers.Conv2D(
        n_classes, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model


def print_members():
    idx = 0
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj) or inspect.isfunction(obj):
            print('    '+str(idx)+': {}'.format(obj))
        idx += 1
    return


if __name__ == '__main__':
    print('----------------------')
    print('Models implementation')
    print('----------------------')
    print('Implemented:')
    print_members()
