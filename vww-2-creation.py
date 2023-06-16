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
    ds_entry,
):  # These arguments are the features of one open images sample
    if tf.reduce_any(tf.equal(tf.constant(68, tf.int64) ,ds_entry["bobjects"]["label"])):  # 68 is the integer label for person
        return ds_entry["image"], 1
    else:
        return ds_entry["image"], 0


example = ds["train"].map(label_person, num_parallel_calls=tf.data.AUTOTUNE)
