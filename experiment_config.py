import time

EXPERIMENT_TIME = time.strftime("%Y_%m_%d-%I_%M_%S_%p")

TARGET_DS = "vww"

MODEL_NAME = f"{TARGET_DS}_mobilenetv1"

WV_DIR = "gs://wake-vision/tensorflow_datasets"
VWW_DIR = "gs://wake-vision/vww"

CHECKPOINT_FILE = f"gs://wake-vision/checkpoints/{EXPERIMENT_TIME}/{MODEL_NAME}/"
SAVE_FILE = f"gs://wake-vision/saved_models/{EXPERIMENT_TIME}/{MODEL_NAME}.keras"

COUNT_PERSON_SAMPLES_TRAIN = 844965  # Number of person samples in the train sdataset. The number of non-person samples are 898077. We will use this number to balance the dataset.
COUNT_PERSON_SAMPLES_VAL = 9973  # There are 31647 non-person samples.
COUNT_PERSON_SAMPLES_TEST = 30226  # There are 95210 non-person samples. The distribution of persons in both the Val and Test set is close to 24% (Val:23.96) (Test:24.09) so we may not need to reduce the size of these.

INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 128
NUM_CLASSES = 2
