import tensorflow as tf
import tensorflow_datasets as tfds

# Config
DATA_DIR = "./dataset/"

ds, info = tfds.load(
    "open_images_v4/200k",
    batch_size=128,
    data_dir=DATA_DIR,
    with_info=True,
    shuffle_files=True,
)

ds["train"].prefetch(tf.data.AUTOTUNE)  # Prefetch the data for improved performance


def label_person(
    bobjects, image, image_filename, objects, objects_trainable
):  # These arguments are the features of one open images sample
    if 68 in bobjects["Label"]:  # 68 is the integer label for person
        return image, 1
