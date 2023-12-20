from train import train
from experiment_config import get_cfg
from cross_validate import cross_val

vww_small_config = get_cfg("vww_small")
vww_small_config.TARGET_DS = "vww"
vww_small_config.MODEL_NAME = "vww_small_mobilenetv1"
vww_small_config.SAVE_FILE = vww_small_config.SAVE_DIR + f"{vww_small_config.MODEL_NAME}.keras"
vww_small_config.MODEL_SIZE = 0.25
vww_small_config.INPUT_SHAPE = (96, 96, 3)

wv_small_config = get_cfg("wv_small")
wv_small_config.TARGET_DS = "wv"
wv_small_config.MODEL_NAME = "wv_small_mobilenetv1_image"
wv_small_config.LABEL_TYPE = "image"
wv_small_config.SAVE_FILE = wv_small_config.SAVE_DIR + f"{wv_small_config.MODEL_NAME}.keras"
wv_small_config.MODEL_SIZE = 0.25
wv_small_config.INPUT_SHAPE = (96, 96, 3)

train(vww_small_config)
train(wv_small_config)

print("Small Model Cross Val Results:")
small_cross_val = cross_val(wv_small_config, vww_small_config)

vww_large_config = get_cfg("vww_large")
vww_large_config.TARGET_DS = "vww"
vww_large_config.EXPERIMENT_NAME = "vww_large"
vww_large_config.MODEL_NAME = "vww_large_mobilenetv1"
vww_large_config.SAVE_FILE = vww_large_config.SAVE_DIR + f"{vww_large_config.MODEL_NAME}.keras"
vww_large_config.MODEL_SIZE = 1.0


wv_large_config = get_cfg("wv_large")
wv_large_config.TARGET_DS = "wv"
wv_large_config.MODEL_NAME = "wv_large_mobilenetv1"
wv_large_config.SAVE_FILE = wv_large_config.SAVE_DIR + f"{wv_large_config.MODEL_NAME}.keras"
wv_large_config.MODEL_SIZE = 1.0

train(vww_large_config)
train(wv_large_config)

print("Large Model Cross Val Results:")
large_cross_val = cross_val(wv_large_config, vww_large_config)

print("Small Model Cross Val Results:")
print(small_cross_val)