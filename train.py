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

model = keras.applications.MobileNet(
    input_shape=cfg.INPUT_SHAPE,
    alpha=0.25,
    weights=None,
    classes=cfg.NUM_CLASSES)

"""
Here's our model summary:
"""

model.summary()

"""
We use the `compile()` method to specify the optimizer, loss function,
and the metrics to monitor. Note that with the JAX and TensorFlow backends,
XLA compilation is turned on by default.
"""

initial_learning_rate = 0.005
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=cfg.CHECKPOINT_FILE,
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


model.fit(
    train, epochs=100, verbose=1, validation_data=val,
    callbacks=[model_checkpoint_callback]
)
score = model.evaluate(test, verbose=1)
print(score)
model.save(cfg.SAVE_FILE)
