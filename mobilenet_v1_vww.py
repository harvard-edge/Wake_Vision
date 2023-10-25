# This file contains a script to load a mobilenetv1 model trained on the visual wake words dataset and report the test accuracy.

import tensorflow as tf

import wake_vision_creation
import mobilenet_v1_dataset_preprocessing as preprocessing

wake_vision = wake_vision_creation.create_wake_vision_ds()

wake_vision_test = preprocessing.mobilenetv1_preprocessing(
    wake_vision["test"], one_hot=True
)

mobilenet_v1_vww = tf.keras.models.load_model("models/mobilenet_v1_vww.h5")

mobilenet_v1_vww.summary()

mobilenet_v1_vww.evaluate(wake_vision_test, verbose=2)

print(mobilenet_v1_vww.get_metrics_result())
