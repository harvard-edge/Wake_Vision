import time
from ml_collections import config_dict

def get_cfg(experiment_name=None):

    cfg = config_dict.ConfigDict()

    cfg.BUCKET_NAME = "gs://wake-vision-storage/"
    cfg.EXPERIMENT_TIME = time.strftime("%Y_%m_%d-%I_%M_%S_%p")
    cfg.EXPERIMENT_NAME = experiment_name + cfg.EXPERIMENT_TIME if experiment_name else cfg.EXPERIMENT_TIME

    cfg.TARGET_DS = "wv"  # Available options are "wv" or "vww"
    cfg.LABEL_TYPE = "bbox"  # Only used for the wake_vision dataset. Specifies whether to use open images image-level labels or bounding boxes. Available options are "image" or "bbox".
    cfg.MIN_BBOX_SIZE = 0.05  # Minimum size of bounding box containing person or subclass for image to be labelled as person. Only works for the wake vision dataset. The visual wake words dataset sets this to 0.05.

    cfg.MODEL_NAME = f"{cfg.TARGET_DS}_mobilenetv1"

    cfg.WV_DIR = f"{cfg.BUCKET_NAME}tensorflow_datasets"
    cfg.VWW_DIR = f"{cfg.BUCKET_NAME}vww"

    cfg.CHECKPOINT_DIR = (
        f"{cfg.BUCKET_NAME}checkpoints/{cfg.EXPERIMENT_NAME}/{cfg.MODEL_NAME}/"
    )
    cfg.SAVE_DIR = f"{cfg.BUCKET_NAME}saved_models/{cfg.EXPERIMENT_NAME}/"
    cfg.SAVE_FILE = cfg.SAVE_DIR + f"{cfg.MODEL_NAME}.keras"


    #TODO recalculate these numbers
    if cfg.LABEL_TYPE == "image":
        cfg.COUNT_PERSON_SAMPLES_TRAIN = 3238953
        cfg.COUNT_PERSON_SAMPLES_VAL = 19311
        cfg.COUNT_PERSON_SAMPLES_TEST = 58288
    cfg.COUNT_PERSON_SAMPLES_TRAIN = 675411#844965  # Number of person samples in the WV train dataset. The number of non-person samples are 898077. We will use this number to balance the dataset.
    cfg.COUNT_PERSON_SAMPLES_VAL = 9106#9973  # There are 31647 non-person samples.
    cfg.COUNT_PERSON_SAMPLES_TEST = 27328#30226  # There are 95210 non-person samples. The distribution of persons in both the Val and Test set is close to 24% (Val:23.96) (Test:24.09) so we may not need to reduce the size of these.

    # Model Config
    cfg.INPUT_SHAPE = (224, 224, 3)
    cfg.NUM_CLASSES = 2
    cfg.MODEL_SIZE = 0.25

    #Train Config
    cfg.STEPS = (10 ** 5) *2
    cfg.VAL_STEPS = cfg.STEPS // 20
    cfg.BATCH_SIZE = 128

    #Learning Rate Config
    cfg.INIT_LR = 0.00001
    cfg.WARMUP_STEPS = 10 ** 3
    cfg.LR = 0.002
    cfg.DECAY_STEPS = cfg.STEPS - cfg.WARMUP_STEPS

    #Weight Decay Config
    cfg.WEIGHT_DECAY = 0.000004

    cfg.SHUFFLE_BUFFER_SIZE = 1024

    return cfg

default_cfg = get_cfg()