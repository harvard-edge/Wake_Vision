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

from wake_vision_loader import get_miaps


def miap_eval(model_cfg):
    model_path = model_cfg.SAVE_FILE
    print("Loading Model:" f"{model_path}")
    model = keras.saving.load_model(model_path)

    miap_ds = get_miaps(model_cfg, batch_size=1)

    female_score = model.evaluate(miap_ds["female"], verbose=1)
    male_score = model.evaluate(miap_ds["male"], verbose=1)
    gender_unknown_score = model.evaluate(miap_ds["gender_unknown"], verbose=1)
    young_score = model.evaluate(miap_ds["young"], verbose=1)
    middle_score = model.evaluate(miap_ds["middle"], verbose=1)
    old_score = model.evaluate(miap_ds["older"], verbose=1)
    age_unknown_score = model.evaluate(miap_ds["age_unknown"], verbose=1)
    no_person_score = model.evaluate(miap_ds["no_person"], verbose=1)

    result = (
        "MIAP Eval Results:"
        f"\n\tPredominantly Female: {female_score[1]}"
        f"\n\tPredominantly Male: {male_score[1]}"
        f"\n\tGender Unknown: {gender_unknown_score[1]}"
        f"\n\tYoung: {young_score[1]}"
        f"\n\tMiddle: {middle_score[1]}"
        f"\n\tOlder: {old_score[1]}"
        f"\n\tAge Unknown: {age_unknown_score[1]}"
        f"\n\tNo Person: {no_person_score[1]}"
    )

    print(result)

    return result


if __name__ == "__main__":
    model_yaml = "gs://wake-vision-storage/saved_models/wv_large2023_12_30-03_29_40_PM/config.yaml"

    with tf.io.gfile.GFile(model_yaml, "r") as fp:
        model_cfg = yaml.unsafe_load(fp)
        model_cfg = config_dict.ConfigDict(model_cfg)

    miap_eval(model_cfg)
