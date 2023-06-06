import tensorflow_datasets as tfds

# Config
DATA_DIR = "./dataset/"

dataset = tfds.load("open_images_v4/200k", batch_size=128, data_dir=DATA_DIR, with_info=True)
