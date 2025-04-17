from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import os

model_name = 'wv_quality_micronets_vww2_50_50_INT8'

input_shape = (50,50,1)
color_mode = 'grayscale'
num_classes = 2

batch_size = 512
epochs = 100
steps_per_epoch = 2438
learning_rate = 0.001

path_to_training_set = '../../datasets/wake_vision/wake_vision/train_quality'
path_to_validation_set = '../../datasets/wake_vision/wake_vision/validation'

inputs = keras.Input(shape=input_shape)
#CONV 3,3,12
x = keras.layers.Conv2D(12, (3,3), padding='same', strides=(2,2))(inputs)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
#IBN 4, 8
x = keras.layers.Conv2D(4, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(8, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#IBN 28, 24
x = keras.layers.Conv2D(28, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(24, (1,1), padding='same')(x)
y = keras.layers.BatchNormalization()(x)
#IBN 28, 20
x = keras.layers.Conv2D(28, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(20, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#padding and add
x = keras.layers.ZeroPadding2D(padding=(0, 2), data_format="channels_first")(x)
x = keras.layers.Add()([x, y])
#IBN 128, 28
x = keras.layers.Conv2D(128, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(28, (1,1), padding='same')(x)
y = keras.layers.BatchNormalization()(x)
#IBN 152, 16
x = keras.layers.Conv2D(152, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(16, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#padding and add
x = keras.layers.ZeroPadding2D(padding=(0, 6), data_format="channels_first")(x)
y = keras.layers.Add()([x, y])
#IBN 152, 28
x = keras.layers.Conv2D(152, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(28, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#add
x = keras.layers.Add()([x, y])
#IBN 152, 24
x = keras.layers.Conv2D(152, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(24, (1,1), padding='same')(x)
y = keras.layers.BatchNormalization()(x)
#IBN 112, 24
x = keras.layers.Conv2D(112, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(24, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#add
y = keras.layers.Add()([x, y])
#IBN 112, 12
x = keras.layers.Conv2D(112, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(12, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#padding and add
x = keras.layers.ZeroPadding2D(padding=(0, 6), data_format="channels_first")(x)
y = keras.layers.Add()([x, y])
#IBN 76, 32
x = keras.layers.Conv2D(76, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(32, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#padding and add
y = keras.layers.ZeroPadding2D(padding=(0, 4), data_format="channels_first")(y)
x = keras.layers.Add()([x, y])
#IBN 76, 16
x = keras.layers.Conv2D(76, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(16, (1,1), padding='same')(x)
y = keras.layers.BatchNormalization()(x)
#IBN 56, 28
x = keras.layers.Conv2D(56, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(28, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#padding and add
y = keras.layers.ZeroPadding2D(padding=(0, 6), data_format="channels_first")(y)
y = keras.layers.Add()([x, y])
#IBN 56, 28
x = keras.layers.Conv2D(56, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(28, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#add
x = keras.layers.Add()([x, y])
#IBN 56, 16
x = keras.layers.Conv2D(56, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(16, (1,1), padding='same')(x)
y = keras.layers.BatchNormalization()(x)
#IBN 96, 16
x = keras.layers.Conv2D(96, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(16, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#add
y = keras.layers.Add()([x, y])
#IBN 96, 16
x = keras.layers.Conv2D(96, (1,1), padding='same')(y)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(16, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#add
x = keras.layers.Add()([x, y])
#IBN 96, 32
x = keras.layers.Conv2D(96, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.DepthwiseConv2D((3,3), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.Conv2D(32, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
#CONV 1,1,128
x = keras.layers.Conv2D(128, (1,1), padding='same')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
#AVG POOLING
x = keras.layers.AveragePooling2D(2)(x)
#CONV 1,1,2
x = keras.layers.Conv2D(2, (1,1), padding='same')(x)
outputs = keras.layers.Reshape((num_classes,))(x)

model = keras.Model(inputs, outputs)

#compile model
opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=opt,
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
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
