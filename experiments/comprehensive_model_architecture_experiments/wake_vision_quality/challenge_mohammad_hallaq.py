from tensorflow_model_optimization.python.core.keras.compat import keras
import tensorflow_model_optimization as tfmot
import tensorflow as tf
import numpy as np
import os

model_name = 'wv_quality_mohammad_hallaq'

input_shape = (80,80,3)
color_mode = 'rgb'
num_classes = 2

batch_size = 512
epochs = 100
steps_per_epoch = 2438
learning_rate = 0.001

path_to_training_set = '../../datasets/wake_vision/wake_vision/train_quality'
path_to_validation_set = '../../datasets/wake_vision/wake_vision/validation'

def build_qat_mobilenetv2(input_shape=input_shape, num_classes=2):
    inputs = keras.Input(shape=input_shape)

    
    x = keras.layers.Conv2D(32, kernel_size=3, strides=2, padding="same", use_bias=False)(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.0)(x)

    
    def inverted_residual_block(x, in_channels, out_channels, expand_channels=0, stride=1):

        residual = x 

        if expand_channels:
            x = keras.layers.Conv2D(expand_channels, kernel_size=1, use_bias=False)(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.ReLU(6.0)(x)

        x = keras.layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.ReLU(6.0)(x)

        x = keras.layers.Conv2D(out_channels, kernel_size=1, use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)

        # *ADD SKIP CONNECTION*
        if stride == 1 and in_channels == out_channels:
            x = keras.layers.Add()([x, residual])  

        return x

    x = inverted_residual_block(x, 32, 16, stride=1)
    x = inverted_residual_block(x, 16, 24, 3, stride=2)
    x = inverted_residual_block(x, 24, 24, 5, stride=1)
    x = inverted_residual_block(x, 24, 32, 5, stride=2)
    x = inverted_residual_block(x, 32, 32, 7, stride=1)
    x = inverted_residual_block(x, 32, 32, 7, stride=1)
    x = inverted_residual_block(x, 32, 64, 7, stride=2)
    x = inverted_residual_block(x, 64, 64, 15, stride=1)
    x = inverted_residual_block(x, 64, 64, 15, stride=1)
    x = inverted_residual_block(x, 64, 64, 15, stride=1)
    x = inverted_residual_block(x, 64, 96, 15, stride=1)
    x = inverted_residual_block(x, 96, 96, 23, stride=1)
    x = inverted_residual_block(x, 96, 96, 23, stride=1)
    x = inverted_residual_block(x, 96, 160, 23, stride=2)
    x = inverted_residual_block(x, 160, 160, 28, stride=1)
    x = inverted_residual_block(x, 160, 160, 28, stride=1)
    x = inverted_residual_block(x, 160, 3, 9, stride=1)

    
    x = keras.layers.Conv2D(38, kernel_size=1, use_bias=False)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.0)(x)

    
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes)(x)  

    model = keras.Model(inputs, outputs)
    return model


model = build_qat_mobilenetv2()

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
