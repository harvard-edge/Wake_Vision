"""
Training Script for Wake Vision and Visual Wake Words Datasets
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

from experiment_config import cfg
from wake_vision_loader import get_wake_vision
from vww_loader import get_vww

from pathlib import Path
import yaml

# TODO fix checkpointing
# with tf.io.gfile.GFile(f'{cfg.CHECKPOINT_DIR}config.yaml', 'w') as fp:
#     yaml.dump(cfg.to_yaml(), fp)


#get data
if cfg.TARGET_DS == "vww":
    train, val, test = get_vww(cfg.BATCH_SIZE)
else:
    train, val, test = get_wake_vision(cfg.BATCH_SIZE)

model = keras.applications.MobileNet(
    input_shape=cfg.INPUT_SHAPE,
    alpha=cfg.MODEL_SIZE,
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

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    cfg.INIT_LR, decay_steps=cfg.DECAY_STEPS, decay_rate=cfg.DECAY_RATE, staircase=True
)

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(),
    optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ],
)

# TODO fix checkpointing
# model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
#     filepath=f"{cfg.CHECKPOINT_DIR}checkpoint.weights.h5",
#     save_weights_only=True,
#     monitor='val_acc',
#     mode='max',
#     save_best_only=True)


model.fit(
    train, epochs=cfg.EPOCHS, verbose=1, validation_data=val,
    # callbacks=[model_checkpoint_callback]
)
score = model.evaluate(test, verbose=1)
print(score)

model.save(cfg.SAVE_FILE)
with open(f'{cfg.SAVE_DIR}config.yaml', 'w') as fp:
    json.dump(cfg.to_yaml(), fp)
