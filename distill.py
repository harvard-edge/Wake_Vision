"""
Distillation Script for Wake Vision and Visual Wake Words Datasets
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
import numpy as np

from experiment_config import default_cfg, get_cfg
from wake_vision_loader import get_wake_vision, get_miaps
from vww_loader import get_vww

import wandb
from wandb.keras import WandbMetricsLogger


def distill(teacher_config, student_cfg=default_cfg):
    wandb.init(
        project="wake-vision",
        name=student_cfg.EXPERIMENT_NAME+"_Distill",
        config=student_cfg,
    )
    if student_cfg.TARGET_DS == "vww":
        train, val, test = get_vww(student_cfg)
    else:
        train, val, test = get_wake_vision(student_cfg)

    student = keras.applications.MobileNetV2(
        input_shape=student_cfg.INPUT_SHAPE,
        alpha=student_cfg.MODEL_SIZE,
        weights=None,
        classes=student_cfg.NUM_CLASSES,
    )

    teacher_path = teacher_config.SAVE_FILE
    print("Loading Teacher:"
          f"{teacher_path}")
    teacher = keras.saving.load_model(teacher_path)
    teacher.trainable = False
    
    print("Student Summary:")
    student.summary()

    print("Teacher Summary:")
    teacher.summary()

    print("Teacher Evaluation:")
    print(teacher.evaluate(val, verbose=1))

    class Distiller(keras.Model):
        def __init__(self, student, **kwargs):
            super().__init__(**kwargs)
            self.student = student

        def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=1.0,
            **kwargs,
        ):
            """Configure the distiller.

            Args:
                optimizer: Keras optimizer for the student weights
                metrics: Keras metrics for evaluation
                student_loss_fn: Loss function of difference between student
                    predictions and ground-truth
                distillation_loss_fn: Loss function of difference between soft
                    student predictions and soft teacher predictions
                alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            """
            super().compile(optimizer=optimizer, metrics=metrics, **kwargs)
            self.student_loss_fn = student_loss_fn
            self.distillation_loss_fn = distillation_loss_fn
            self.alpha = alpha

        def compute_loss(self, x, y, y_pred, sample_weight=None, allow_empty=False):
            teacher_pred = teacher(x, training=False)

            student_loss = self.student_loss_fn(y, y_pred)
            distillation_loss = self.distillation_loss_fn(teacher_pred, y_pred)

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

            return loss

        def call(self, x, training=False):
            return self.student(x, training=training)

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        student_cfg.INIT_LR,
        decay_steps=student_cfg.DECAY_STEPS,
        alpha=0.0,
        warmup_target=student_cfg.LR,
        warmup_steps=student_cfg.WARMUP_STEPS,
    )

    distiller = Distiller(student=student)
    distiller.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=student_cfg.WEIGHT_DECAY
        ),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        student_loss_fn=keras.losses.SparseCategoricalCrossentropy(),
        distillation_loss_fn=keras.losses.KLDivergence(),
        alpha=0.0,
    )

    

    callbacks = [WandbMetricsLogger()]

    # Train for a fixed number of steps, validating every
    distiller.fit(
        train, epochs=(student_cfg.STEPS//student_cfg.VAL_STEPS), steps_per_epoch=student_cfg.VAL_STEPS, validation_data=val,
        callbacks=callbacks,
    )
    score = distiller.evaluate(test, verbose=1)
    print(score)

    student.save(student_cfg.SAVE_FILE)
    with tf.io.gfile.GFile(f"{student_cfg.SAVE_DIR}config.yaml", "w") as fp:
        student_cfg.to_yaml(stream=fp)

    # return path to saved model, to be evaluated
    wandb.finish()
    return student_cfg.SAVE_FILE


if __name__ == "__main__":
    import argparse
    import yaml
    from ml_collections import config_dict

    teacher_name = "2024_02_05-03_28_30_PM"

    cfg = get_cfg()

    parser = argparse.ArgumentParser()
    parser.add_argument("--target_ds", type=str, default=cfg.TARGET_DS)
    parser.add_argument("--model_size", type=float, default=cfg.MODEL_SIZE)
    parser.add_argument(
        "--input_size", type=str, default=",".join(map(str, cfg.INPUT_SHAPE))
    )
    parser.add_argument("--lr", type=float, default=cfg.LR)
    parser.add_argument("--bs", type=int, default=cfg.BATCH_SIZE)


    args = parser.parse_args()
    cfg.TARGET_DS = args.target_ds
    cfg.MODEL_SIZE = args.model_size
    cfg.INPUT_SHAPE = tuple(map(int, args.input_size.split(",")))
    cfg.LR = args.lr
    cfg.BATCH_SIZE = args.bs

    print("teacher_name:", teacher_name)
    teacher_yaml = "gs://wake-vision-storage/saved_models/" + teacher_name + "/config.yaml"
    with tf.io.gfile.GFile(teacher_yaml, 'r') as fp:
        teacher_cfg = yaml.unsafe_load(fp)
        teacher_cfg = config_dict.ConfigDict(teacher_cfg)

    distill(teacher_cfg, cfg)