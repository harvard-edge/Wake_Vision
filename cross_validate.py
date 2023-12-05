# %%
import numpy as np
import os
import pandas as pd

os.environ["KERAS_BACKEND"] = "jax"

# Note that keras_core should only be imported after the backend
# has been configured. The backend cannot be changed once the
# package is imported.
import keras_core as keras

import tensorflow as tf
import tensorflow_datasets as tfds

from wake_vision_loader import get_wake_vision
from vww_loader import get_vww

vww_model = keras.saving.load_model("gs://wake-vision-storage/saved_models/2023_11_09-08_46_32_PM/vww_mobilenetv1.keras")
wv_model = keras.saving.load_model("gs://wake-vision-storage/saved_models/2023_11_08-06_42_41_PM/wv_mobilenetv1.keras")


_, _, vww_test = get_vww()
_, _, wv_test = get_wake_vision()

# %%

vww_model_vww_test_score = vww_model.evaluate(vww_test, verbose=1)
vww_model_wv_test_score = vww_model.evaluate(wv_test, verbose=1)

wv_model_vww_test_score = wv_model.evaluate(vww_test, verbose=1)
wv_model_wv_test_score = wv_model.evaluate(wv_test, verbose=1)

# %%


cross_val = np.array([
    ["","Train", "VWW", "WV"],
    ["Test","VWW", vww_model_vww_test_score[1], wv_model_vww_test_score[1]],
    ["Test", "WV", vww_model_wv_test_score[1], wv_model_wv_test_score[1]],
    ["","", "", ""],
],dtype=object)

#create df
cross_val = pd.DataFrame(cross_val[1:,1:], index=cross_val[1:,0], columns=cross_val[0,1:])

print("Cross Val Results:")
print(cross_val)

# %%
