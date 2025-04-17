from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import os

model_name = 'wv_quality_mcunet-5fps_vww'

input_shape = (80,80,3)
color_mode = 'rgb'
num_classes = 2

batch_size = 512
epochs = 100
steps_per_epoch = 2438
learning_rate = 0.001

path_to_training_set = '../../datasets/wake_vision/wake_vision/train_quality'
path_to_validation_set = '../../datasets/wake_vision/wake_vision/validation'

inputs = keras.Input(shape=input_shape)
#
x = keras.layers.ZeroPadding2D(padding=(1, 1))(inputs)
x = keras.layers.Conv2D(16, (3,3), padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(8, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(16, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = keras.layers.DepthwiseConv2D((7,7),  padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1, 1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(120, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(24, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(144, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(3, 3))(x)
x = keras.layers.DepthwiseConv2D((7,7),  padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(240, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(40, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(240, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(192, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(48, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(240, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(2,2))(x)
x = keras.layers.DepthwiseConv2D((5,5),  padding='valid', strides=(2,2))(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
y = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(480, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
y = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(y)
x = keras.layers.Conv2D(384, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(1,1))(x)
x = keras.layers.DepthwiseConv2D((3,3),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(96, (1,1), padding='valid')(x)
# add
x = keras.layers.Add()([x, y])
#
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(288, (1,1), padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(3,3))(x)
x = keras.layers.DepthwiseConv2D((7,7),  padding='valid')(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU(max_value=6.0)(x)
x = keras.layers.ZeroPadding2D(padding=(0, 0))(x)
x = keras.layers.Conv2D(160, (1,1), padding='valid')(x)
#
x = keras.layers.AveragePooling2D(2)(x)
x = keras.layers.Conv2D(2, (1,1), padding='valid')(x)
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
