"""
Training Script for Wake Vision and Visual Wake Words Datasets
"""
import numpy as np
import os

os.environ["KERAS_BACKEND"] = "jax"

# Note that keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras

import tensorflow as tf
import tensorflow_datasets as tfds

from experiment_config import default_cfg, get_cfg
from wake_vision_loader import get_wake_vision
from vww_loader import get_vww

import wandb
from wandb.keras import WandbMetricsLogger

def train(cfg=default_cfg, distance_eval=False):

    wandb.init(project="wake-vision", config=cfg)

    # TODO fix checkpointing
    # with tf.io.gfile.GFile(f'{cfg.CHECKPOINT_DIR}config.yaml', 'w') as fp:
    #     yaml.dump(cfg.to_yaml(), fp)

    if cfg.TARGET_DS == "vww":
        train, val, test = get_vww(cfg)
    else:
        train, val, test = get_wake_vision(cfg)

    model = keras.applications.MobileNetV2(
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
    lr_schedule = keras.optimizers.schedules.CosineDecay(
        cfg.INIT_LR, decay_steps=cfg.DECAY_STEPS, alpha=0.0,
        warmup_target=cfg.LR, warmup_steps=cfg.WARMUP_STEPS
    )

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.AdamW(learning_rate=lr_schedule,
                                          weight_decay=cfg.WEIGHT_DECAY,),
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
    callbacks = [WandbMetricsLogger()]

    #Distance Eval on each epoch
    if distance_eval:
        from wake_vision_loader import get_distance_eval
        class DistanceEvalCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                distance_ds = get_distance_eval(cfg)

                near_score = self.model.evaluate(distance_ds["near"], verbose=1)
                mid_score = self.model.evaluate(distance_ds["mid"], verbose=1)
                far_score = self.model.evaluate(distance_ds["far"], verbose=1)
                no_person_score = self.model.evaluate(distance_ds["no_person"], verbose=1)
                result = ("Distace Eval Results:"
                    f"\n\tNear: {near_score[1]}"
                    f"\n\tMid: {mid_score[1]}"
                    f"\n\tFar: {far_score[1]}"
                    f"\n\tNo Person: {no_person_score[1]}")
                print(result)
        
        callbacks.append(DistanceEvalCallback())

    #Train for a fixed number of steps, validating every
    model.fit(
        train, epochs=(cfg.STEPS//cfg.VAL_STEPS), steps_per_epoch=cfg.VAL_STEPS, validation_data=val,
        callbacks=callbacks,
    )
    score = model.evaluate(test, verbose=1)
    print(score)

    model.save(cfg.SAVE_FILE)
    with tf.io.gfile.GFile(f'{cfg.SAVE_DIR}config.yaml', 'w') as fp:
        cfg.to_yaml(stream=fp)


    #return path to saved model, to be evaluated
    wandb.finish()
    return cfg.SAVE_FILE


if __name__ == "__main__":
    import argparse

    cfg = get_cfg()

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ds", type=str, default=cfg.TARGET_DS)
    parser.add_argument("--model_size", type=float, default=cfg.MODEL_SIZE)
    parser.add_argument("--input_size", type=str, default=','.join(map(str, cfg.INPUT_SHAPE)))

    args = parser.parse_args()
    cfg.TARGET_DS = args.target_ds
    cfg.MODEL_SIZE = args.model_size
    cfg.INPUT_SHAPE = tuple(map(int, args.input_size.split(",")))

    train(cfg)