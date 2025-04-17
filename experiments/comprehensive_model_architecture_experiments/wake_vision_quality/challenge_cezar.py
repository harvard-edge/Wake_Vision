from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import os

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, DepthwiseConv2D, Conv2D, BatchNormalization,
    Activation, Add, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.models import Model

model_name = 'wv_quality_cezar'

input_shape = (50,50,3)
color_mode = 'rgb'
num_classes = 2

batch_size = 512
epochs = 100
steps_per_epoch = 2438
learning_rate = 0.001

path_to_training_set = '../../datasets/wake_vision/wake_vision/train_quality'
path_to_validation_set = '../../datasets/wake_vision/wake_vision/validation'

def identity_block(x, filters, kernel_size=3):
    """An identity block using depthwise separable convolutions."""
    shortcut = x

    # First depthwise separable block:
    x = DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Second depthwise separable block:
    x = DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Add the shortcut (assumed to have same dimensions) and activate
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def conv_block(x, filters, kernel_size=3, strides=2):
    """A convolutional block using depthwise separable convolutions for downsampling."""
    shortcut = x

    # Main branch: first separable block with downsampling via strides.
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Second separable block without additional downsampling.
    x = DepthwiseConv2D(kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Shortcut branch: use a 1x1 convolution to match dimensions.
    shortcut = Conv2D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
    shortcut = BatchNormalization()(shortcut)

    # Add shortcut and main branch, then apply activation.
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def ResNetLiteDepthwise(input_shape, num_classes):
    """Constructs a lightweight ResNet-Lite model using depthwise separable convolutions.
    
    Args:
        input_shape: Tuple of the input dimensions, e.g. (50, 50, 3) for Color images.
        num_classes: Number of output classes.
    
    Returns:
        A Keras Model instance.
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution (not separable) to lift input channels.
    x = Conv2D(32, 3, strides=1, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # First residual stage: no spatial downsampling; use identity block.
    x = identity_block(x, filters=32)
    
    # Second residual stage: downsample and increase filter count.
    x = conv_block(x, filters=64, strides=2)
    x = identity_block(x, filters=64)
    
    # Third residual stage: further downsampling.
    #x = conv_block(x, filters=128, strides=2)
    #x = identity_block(x, filters=128)
    
    # Global average pooling and the classification head.
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name="ResNetLite_Depthwise")
    return model


model = ResNetLiteDepthwise(input_shape=(50, 50, 3), num_classes=2)

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt,
    loss='categorical_crossentropy',
    metrics=['accuracy'])

#load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    directory= path_to_training_set,
    labels='inferred',
    label_mode='categorical',
    color_mode=color_mode,
    batch_size=batch_size,
    image_size=input_shape[0:2],
    shuffle=True,
    seed=11
)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    directory= path_to_validation_set,
    labels='inferred',
    label_mode='categorical',
    color_mode=color_mode,
    batch_size=batch_size,
    image_size=input_shape[0:2],
    shuffle=True,
    seed=11
)

#data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2)])

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath= model_name + ".tf",
    monitor='val_accuracy',
    mode='max', save_best_only=True)

model.fit(train_ds.repeat(), epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=validation_ds, callbacks=[model_checkpoint_callback])
 
model = tf.keras.models.load_model(model_name + ".tf")

def representative_dataset():
    for data in train_ds.rebatch(1).take(150) :
        yield [tf.dtypes.cast(data[0], tf.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8
tflite_quant_model = converter.convert()

with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_quant_model)

