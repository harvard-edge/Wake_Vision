import numpy as np
import os
import pandas as pd
from ml_collections import config_dict
import yaml

os.environ["KERAS_BACKEND"] = "jax"

# Note that keras should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras

import tensorflow as tf
import tensorflow_datasets as tfds

from wake_vision_loader import get_distance_eval


def distance_eval(model_cfg):
    model_path = model_cfg.SAVE_FILE
    print("Loading Model:" f"{model_path}")
    model = keras.saving.load_model(model_path)

    distance_ds = get_distance_eval(model_cfg)

    near_score = model.evaluate(distance_ds["near"], verbose=1)
    mid_score = model.evaluate(distance_ds["mid"], verbose=1)
    far_score = model.evaluate(distance_ds["far"], verbose=1)
    no_person_score = model.evaluate(distance_ds["no_person"], verbose=1)

    result = (
        "Distace Eval Results:"
        f"\n\tNear: {near_score[1]}"
        f"\n\tMid: {mid_score[1]}"
        f"\n\tFar: {far_score[1]}"
        f"\n\tNo Person: {no_person_score[1]}"
    )

    print(result)

    return result


if __name__ == "__main__":
    model_yaml = "gs://wake-vision-storage/saved_models/wv_large2023_12_30-03_29_40_PM/config.yaml"

    with tf.io.gfile.GFile(model_yaml, "r") as fp:
        model_cfg = yaml.unsafe_load(fp)
        model_cfg = config_dict.ConfigDict(model_cfg)

    distance_eval(model_cfg)
