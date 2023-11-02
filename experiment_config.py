import time
from ml_collections import config_dict


cfg = config_dict.ConfigDict()
cfg.EXPERIMENT_TIME = time.strftime("%Y_%m_%d-%I_%M_%S_%p")

cfg.TARGET_DS = "wv"

cfg.MODEL_NAME = f"{cfg.TARGET_DS}_mobilenetv1"

cfg.WV_DIR = "gs://wake-vision/tensorflow_datasets"
cfg.VWW_DIR = "gs://wake-vision/vww"

cfg.CHECKPOINT_DIR = f"gs://wake-vision/checkpoints/{cfg.EXPERIMENT_TIME}/{cfg.MODEL_NAME}/"
cfg.SAVE_DIR = f"gs://wake-vision/saved_models/{cfg.EXPERIMENT_TIME}/"
cfg.SAVE_FILE = cfg.SAVE_DIR+f"{cfg.MODEL_NAME}.keras"

cfg.COUNT_PERSON_SAMPLES_TRAIN = 844965  # Number of person samples in the train sdataset. The number of non-person samples are 898077. We will use this number to balance the dataset.
cfg.COUNT_PERSON_SAMPLES_VAL = 9973  # There are 31647 non-person samples.
cfg.COUNT_PERSON_SAMPLES_TEST = 30226  # There are 95210 non-person samples. The distribution of persons in both the Val and Test set is close to 24% (Val:23.96) (Test:24.09) so we may not need to reduce the size of these.

#Model Config
cfg.INPUT_SHAPE = (224, 224, 3)
cfg.NUM_CLASSES = 2
cfg.MODEL_SIZE = 0.25

#Train Config
cfg.EPOCHS = 100
cfg.BATCH_SIZE = 128
cfg.INIT_LR = 0.001
cfg.DECAY_STEPS = 10000
cfg.DECAY_RATE = 0.96
