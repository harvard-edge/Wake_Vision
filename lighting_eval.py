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

from wake_vision_loader import get_lighting


def lighting_eval(model_cfg):
    model_path = model_cfg.SAVE_FILE
    print("Loading Model:" f"{model_path}")
    model = keras.saving.load_model(model_path)

    lighting_ds = get_lighting(model_cfg, batch_size=1)

    dark_score = model.evaluate(lighting_ds["dark"], verbose=1)
    normal_light_score = model.evaluate(lighting_ds["normal_light"], verbose=1)
    bright_score = model.evaluate(lighting_ds["bright"], verbose=1)

    result = (
        "Lighting Eval Results:"
        f"\n\tDark: {dark_score[1]}"
        f"\n\tDim: {normal_light_score[1]}"
        f"\n\tBright: {bright_score[1]}"
    )

    print(result)

    return result


if __name__ == "__main__":
    model_yaml = "gs://wake-vision-storage/saved_models/wv_large2023_12_30-03_29_40_PM/config.yaml"

    with tf.io.gfile.GFile(model_yaml, "r") as fp:
        model_cfg = yaml.unsafe_load(fp)
        model_cfg = config_dict.ConfigDict(model_cfg)

    lighting_eval(model_cfg)
