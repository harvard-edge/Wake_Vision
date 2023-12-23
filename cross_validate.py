# %%
import numpy as np
import os
import pandas as pd
from ml_collections import config_dict
import yaml

os.environ["KERAS_BACKEND"] = "jax"

# Note that keras_core should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras_core as keras

import tensorflow as tf
import tensorflow_datasets as tfds

from wake_vision_loader import get_wake_vision
from vww_loader import get_vww

def cross_val(wv_model_cfg, vww_model_cfg):
    wv_model_path = wv_model_cfg.SAVE_FILE
    vww_model_path = vww_model_cfg.SAVE_FILE
    print("Loading Models:"
          f"\n\tWake Vision Model: {wv_model_path}"
          f"\n\tVisual Wake Words Model: {vww_model_path}")
    wv_model = keras.saving.load_model(wv_model_path)
    vww_model = keras.saving.load_model(vww_model_path)

    _, _, vww_test = get_vww(vww_model_cfg)
    _, _, wv_test = get_wake_vision(wv_model_cfg)

    vww_model_vww_test_score = vww_model.evaluate(vww_test, verbose=1)
    vww_model_wv_test_score = vww_model.evaluate(wv_test, verbose=1)

    wv_model_vww_test_score = wv_model.evaluate(vww_test, verbose=1)
    wv_model_wv_test_score = wv_model.evaluate(wv_test, verbose=1)

    cross_val = np.array([
        ["","Train", vww_model_cfg.MODEL_NAME, wv_model_cfg.MODEL_NAME],
        ["Test","VWW", vww_model_vww_test_score[1], wv_model_vww_test_score[1]],
        ["Test", "WV", vww_model_wv_test_score[1], wv_model_wv_test_score[1]],
        ["","", "", ""],
    ],dtype=object)

    #create df
    cross_val = pd.DataFrame(cross_val[1:,1:], index=cross_val[1:,0], columns=cross_val[0,1:])

    print("Cross Val Results:")
    print(cross_val)

    return cross_val

if __name__ == "__main__":
    wv_yaml = "gs://wake-vision-storage/saved_models/wv_small2023_12_19-09_52_03_PM/config.yaml"
    vww_yaml = "gs://wake-vision-storage/saved_models/vww_small2023_12_19-01_02_00_AM/config.yaml"

    with tf.io.gfile.GFile(wv_yaml, 'r') as fp:
        wv_cfg = yaml.unsafe_load(fp)
        print(type(wv_cfg))
        wv_cfg = config_dict.ConfigDict(wv_cfg)

    with tf.io.gfile.GFile(vww_yaml, 'r') as fp:
        vww_cfg = yaml.unsafe_load(fp)
        vww_cfg = config_dict.ConfigDict(vww_cfg)


    cross_val(wv_cfg, vww_cfg)
