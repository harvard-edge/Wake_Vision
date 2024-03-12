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

from wake_vision_loader import get_wake_vision


def cross_val(image_level_model_cfg, bbox_model_cfg):
    image_level_model_path = image_level_model_cfg.SAVE_FILE
    bbox_model_path = bbox_model_cfg.SAVE_FILE
    print(
        "Loading Models:"
        f"\n\tImage Level Label Trained Model: {image_level_model_path}"
        f"\n\tBounding Boxes Trained Model: {bbox_model_path}"
    )
    image_level_model = keras.saving.load_model(image_level_model_path)
    bbox_model = keras.saving.load_model(bbox_model_path)

    _, _, wv_test = get_wake_vision(image_level_model_cfg)

    bbox_test_score = bbox_model.evaluate(wv_test, verbose=1)
    image_level_test_score = image_level_model.evaluate(wv_test, verbose=1)

    cross_val = np.array(
        [
            ["Train", "Image Level Label Model", "Bounding Box Model"],
            ["Test", image_level_test_score[1], bbox_test_score[1]],
            ["", "", ""],
        ],
        dtype=object,
    )

    # create df
    cross_val = pd.DataFrame(
        cross_val[1:, 1:], index=cross_val[1:, 0], columns=cross_val[0, 1:]
    )

    print("Cross Val Results:")
    print(cross_val)

    return cross_val


if __name__ == "__main__":

    image_level_yaml = "gs://wake-vision-storage/saved_models/image_label2024_02_16-11_44_10_PM/config.yaml"
    bbox_yaml = "gs://wake-vision-storage/saved_models/baseline_bbox_training_run2024_02_19-04_05_02_PM/config.yaml"

    with tf.io.gfile.GFile(image_level_yaml, "r") as fp:
        wv_cfg = yaml.unsafe_load(fp)
        print(type(wv_cfg))
        wv_cfg = config_dict.ConfigDict(wv_cfg)

    with tf.io.gfile.GFile(bbox_yaml, "r") as fp:
        vww_cfg = yaml.unsafe_load(fp)
        vww_cfg = config_dict.ConfigDict(vww_cfg)

    cross_val(wv_cfg, vww_cfg)
