import keras
from keras.models import Model
from keras.layers import Input, Dense, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from keras.regularizers import l2
import numpy as np

#pulled directly from the MLPerf Tiny benchmark suite
def resnet_mlperf(input_shape, num_classes):
    # Resnet parameters

    num_filters = 16 # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = MaxPooling2D(pool_size=(2, 2))(x) # uncomment this for official resnet model


    # First stack

    # Weight layers
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Overall residual, connect weight layer and identity paths
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Second stack

    # Weight layers
    num_filters = 32 # Filters need to be double for each stack
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Third stack

    # Weight layers
    num_filters = 64
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Fourth stack.
    # While the paper uses four stacks, for cifar10 that leads to a large increase in complexity for minor benefits
    # Uncomments to use it

#    # Weight layers
#    num_filters = 128
#    y = Conv2D(num_filters,
#                  kernel_size=3,
#                  strides=2,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(x)
#    y = BatchNormalization()(y)
#    y = Activation('relu')(y)
#    y = Conv2D(num_filters,
#                  kernel_size=3,
#                  strides=1,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(y)
#    y = BatchNormalization()(y)
#
#    # Adjust for change in dimension due to stride in identity
#    x = Conv2D(num_filters,
#                  kernel_size=1,
#                  strides=2,
#                  padding='same',
#                  kernel_initializer='he_normal',
#                  kernel_regularizer=l2(1e-4))(x)
#
#    # Overall residual, connect weight layer and identity paths
#    x = tf.keras.layers.add([x, y])
#    x = Activation('relu')(x)


    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_scaled(input_shape, num_classes, scale=2):
    # Resnet parameters

    num_filters = 16 * scale # this should be 64 for an official resnet model

    # Input layer, change kernel size to 7x7 and strides to 2 for an official resnet
    inputs = Input(shape=input_shape)
    x = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)


    # First stack

    # Weight layers
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Overall residual, connect weight layer and identity paths
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Second stack

    # Weight layers
    num_filters = 32 * scale # Filters need to be double for each stack
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Third stack

    # Weight layers
    num_filters = 64 * scale
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                  kernel_size=3,
                  strides=1,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                  kernel_size=1,
                  strides=2,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Fourth stack.

    # Weight layers
    num_filters = 128 * scale
    y = Conv2D(num_filters,
                    kernel_size=3,
                    strides=2,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4))(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(num_filters,
                    kernel_size=3,
                    strides=1,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4))(y)
    y = BatchNormalization()(y)

    # Adjust for change in dimension due to stride in identity
    x = Conv2D(num_filters,
                    kernel_size=1,
                    strides=2,
                    padding='same',
                    kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-4))(x)

    # Overall residual, connect weight layer and identity paths
    x = keras.layers.add([x, y])
    x = Activation('relu')(x)


    # Final classification layer.
    pool_size = int(np.amin(x.shape[1:3]))
    x = AveragePooling2D(pool_size=pool_size)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def handle_block_names(stage, block):
    name_base = 'stage{}_unit{}_'.format(stage + 1, block + 1)
    conv_name = name_base + 'conv'
    bn_name = name_base + 'bn'
    relu_name = name_base + 'relu'
    sc_name = name_base + 'sc'
    return conv_name, bn_name, relu_name, sc_name

def get_conv_params(**params):
    default_conv_params = {
        'kernel_initializer': 'he_uniform',
        'use_bias': False,
        'padding': 'valid',
    }
    default_conv_params.update(params)
    return default_conv_params


def get_bn_params(**params):
    default_bn_params = {
        'momentum': 0.99,
        'epsilon': 2e-5,
        'center': True,
        'scale': True,
    }
    default_bn_params.update(params)
    return default_bn_params

def ResidualBlock(filters, stage, block, strides=(1, 1), cut='pre'):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        cut: one of 'pre', 'post'. used to decide where skip connection is taken
    # Returns
        Output tensor for the block.
    """

    def layer(input_tensor):

        # get params and names of layers
        conv_params = get_conv_params()
        bn_params = get_bn_params()
        conv_name, bn_name, relu_name, sc_name = handle_block_names(stage, block)

        x = keras.layers.BatchNormalization(name=bn_name + '1', **bn_params)(input_tensor)
        x = keras.layers.Activation('relu', name=relu_name + '1')(x)

        # defining shortcut connection
        if cut == 'pre':
            shortcut = input_tensor
        elif cut == 'post':
            shortcut = keras.layers.Conv2D(filters, (1, 1), name=sc_name, strides=strides, **conv_params)(x)
        else:
            raise ValueError('Cut type not in ["pre", "post"]')

        # continue with convolution layers
        x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = keras.layers.Conv2D(filters, (3, 3), strides=strides, name=conv_name + '1', **conv_params)(x)

        x = keras.layers.BatchNormalization(name=bn_name + '2', **bn_params)(x)
        x = keras.layers.Activation('relu', name=relu_name + '2')(x)
        x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
        x = keras.layers.Conv2D(filters, (3, 3), name=conv_name + '2', **conv_params)(x)

        # add residual connection
        x = keras.layers.Add()([x, shortcut])
        return x

    return layer


def resnet_builder(input_shape, num_classes, layers=[2,2,2,2]):
    
    inputs = Input(shape=input_shape)
    init_filters = 64

    # resnet bottom
    x = keras.layers.BatchNormalization(name='bn_data')(inputs)
    x = keras.layers.ZeroPadding2D(padding=(3, 3))(x)
    x = keras.layers.Conv2D(init_filters, (7, 7), strides=(2, 2), name='conv0')(x)
    x = keras.layers.BatchNormalization(name='bn0')(x)
    x = keras.layers.Activation('relu', name='relu0')(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='valid', name='pooling0')(x)

    # resnet body
    for stage, rep in enumerate(layers):
        for block in range(rep):

            filters = init_filters * (2 ** stage)

            # first block of first stage without strides because we have maxpooling before
            if block == 0 and stage == 0:
                x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                  cut='post')(x)

            elif block == 0:
                x = ResidualBlock(filters, stage, block, strides=(2, 2),
                                  cut='post')(x)

            else:
                x = ResidualBlock(filters, stage, block, strides=(1, 1),
                                  cut='pre')(x)

    x = keras.layers.BatchNormalization(name='bn1')(x)
    x = keras.layers.Activation('relu', name='relu1')(x)

    # resnet top
    x = keras.layers.GlobalAveragePooling2D(name='pool1')(x)
    x = keras.layers.Dense(num_classes, name='fc1')(x)
    x = keras.layers.Activation('softmax', name='softmax')(x)
    
    model = Model(inputs=inputs, outputs=x)
    return model

def resnet8(input_shape, num_classes):
    return resnet_builder(input_shape, num_classes, layers=[1,1,1,1])

def resnet18(input_shape, num_classes):
    return resnet_builder(input_shape, num_classes, layers=[2,2,2,2])

def resnet34(input_shape, num_classes):
    return resnet_builder(input_shape, num_classes, layers=[3, 4, 6, 3])