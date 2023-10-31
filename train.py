"""
Title: Getting started with Keras Core
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2023/07/10
Last modified: 2023/07/10
Description: First contact with the new multi-backend Keras.
Accelerator: GPU
"""
"""
## Introduction

Keras Core is a full implementation of the Keras API that
works with TensorFlow, JAX, and PyTorch interchangeably.
This notebook will walk you through key Keras Core workflows.

First, let's install Keras Core:
"""

"""shell
pip install -q keras-core
"""

"""
## Setup

We're going to be using the JAX backend here -- but you can
edit the string below to `"tensorflow"` or `"torch"` and hit
"Restart runtime", and the whole notebook will run just the same!
This entire guide is backend-agnostic.
"""
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

# Note that keras_core should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras_core as keras

import tensorflow as tf
import tensorflow_datasets as tfds

import experiment_config as cfg
from wake_vision_loader import get_wake_vision
from vww_loader import get_vww


#get data
if cfg.TARGET_DS == "vww":
    train, val, test = get_vww(cfg.BATCH_SIZE)
else:
    train, val, test = get_wake_vision(cfg.BATCH_SIZE)

model = keras.Sequential(
    [
        keras.layers.Input(shape=cfg.INPUT_SHAPE),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(cfg.NUM_CLASSES, activation="softmax"),
    ]
)

"""
Here's our model summary:
"""

model.summary()

"""
We use the `compile()` method to specify the optimizer, loss function,
and the metrics to monitor. Note that with the JAX and TensorFlow backends,
XLA compilation is turned on by default.
"""

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)


model.fit(
    train, epochs=10, verbose=1, steps_per_epoch=(cfg.COUNT_PERSON_SAMPLES_TRAIN//cfg.BATCH_SIZE), validation_data=val
)
score = model.evaluate(test, verbose=0)
print(score)
model.save(cfg.SAVE_FILE)
