from train import train
from experiment_config import get_cfg
from distance_val import distance_val

ds = "wv"

for image_size in [32, 96, 128, 256, 384]:
    cfg = get_cfg(f"{ds}_small_{image_size}x{image_size}_min_bbox_size_0.1")
    cfg.TARGET_DS = ds
    cfg.MODEL_NAME = f"{ds}_small_mobilenetv2"
    cfg.INPUT_SHAPE = (image_size, image_size, 3)
    cfg.SAVE_FILE = cfg.SAVE_DIR + f"{cfg.MODEL_NAME}.keras"
    cfg.MODEL_SIZE = 0.25
    cfg.MIN_BBOX_SIZE = 0.1
    cfg.COUNT_PERSON_SAMPLES_TRAIN = 620029
    print(f"Training {ds} Small {image_size}x{image_size}:")
    train(cfg)
    distance_val(cfg)
