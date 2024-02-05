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

from wake_vision_loader import get_distance_eval, get_wake_vision

def distance_val(model_cfg):
    model_path = model_cfg.SAVE_FILE
    print("Loading Model:"
          f"{model_path}")
    model = keras.saving.load_model(model_path)

    dist_cfg = model_cfg.copy_and_resolve_references()
    dist_cfg.MIN_BBOX_SIZE = 0.05

    _, _, wv_test = get_wake_vision(dist_cfg)

    wv_test_score = model.evaluate(wv_test, verbose=1)

    print(f"Wake Vision Test Score: {wv_test_score[1]}")

    distance_ds = get_distance_eval(dist_cfg)

    near_score = model.evaluate(distance_ds["near"], verbose=1)
    mid_score = model.evaluate(distance_ds["mid"], verbose=1)
    far_score = model.evaluate(distance_ds["far"], verbose=1)
    no_person_score = model.evaluate(distance_ds["no_person"], verbose=1)

    

    result = ("Distace Eval Results:"
        f"\n\tNear: {near_score[1]}"
        f"\n\tMid: {mid_score[1]}"
        f"\n\tFar: {far_score[1]}"
        f"\n\tNo Person: {no_person_score[1]}")
    
    print(result)

    return result

if __name__ == "__main__":
    experiment_names = [
    # "wv_small_32x322024_01_03-09_51_58_PM/",
    # "wv_small_96x962024_01_04-08_38_04_AM/",
    # "wv_small_128x1282024_01_05-12_05_07_AM/",
    # "wv_small_256x2562024_01_05-12_50_55_PM/",
    # "wv_small_384x3842024_01_06-02_25_46_AM/",
    # "wv_small_32x32_min_bbox_size_0.12024_01_17-07_26_56_PM/",
    # "wv_small_96x96_min_bbox_size_0.12024_01_18-03_36_26_PM/",
    # "wv_small_128x128_min_bbox_size_0.12024_01_19-08_25_24_PM/",
    # "wv_small_256x256_min_bbox_size_0.12024_01_20-07_03_01_AM/",
    # "wv_small_384x384_min_bbox_size_0.12024_01_21-04_14_30_AM/",
    "wv_small_32x32_min_bbox_size_0.12024_01_24-02_01_59_PM/",
    "wv_small_96x96_min_bbox_size_0.12024_01_25-12_51_00_AM/",
    "wv_small_128x128_min_bbox_size_0.12024_01_25-12_43_59_PM/",
    "wv_small_256x256_min_bbox_size_0.12024_01_26-03_19_42_AM/",
    "wv_small_384x384_min_bbox_size_0.12024_01_26-02_56_05_PM",
    ]
    for model in experiment_names:
        print(model)
        model_yaml = "gs://wake-vision-storage/saved_models/" + model + "config.yaml"
        with tf.io.gfile.GFile(model_yaml, 'r') as fp:
            model_cfg = yaml.unsafe_load(fp)
            model_cfg = config_dict.ConfigDict(model_cfg)

        distance_val(model_cfg)