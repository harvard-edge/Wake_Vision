import tensorflow as tf
import tensorflow_datasets as tfds

# Config
DATA_DIR = "./dataset/"

ds = tfds.load(
    "open_images_v4/200k",
    data_dir=DATA_DIR,
    shuffle_files=True,
)


def label_person(
    ds_entry,
):
    if tf.reduce_any(
        tf.equal(tf.constant(1208, tf.int64), ds_entry["objects"]["label"])
    ):  # 1208 is the integer value for the image level label for person
        ds_entry["person"] = 1
    else:
        ds_entry["person"] = 0


ds["train"] = ds["train"].map(label_person, num_parallel_calls=tf.data.AUTOTUNE)
