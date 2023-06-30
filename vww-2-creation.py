import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# Config
DATA_DIR = "./dataset/"
COUNT_PERSON_SAMPLES = 844965  # Number of person samples in the dataset. The number of non-person samples are 898077. We will use this number to balance the dataset.

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
    return ds_entry


def person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 1)


def non_person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 0)


ds["train"] = ds["train"].map(label_person, num_parallel_calls=tf.data.AUTOTUNE)

person_ds = ds["train"].filter(person_filter)
non_person_ds = ds["train"].filter(non_person_filter)

non_person_ds = non_person_ds.take(COUNT_PERSON_SAMPLES)
person_ds = person_ds.take(COUNT_PERSON_SAMPLES)

ds["train"] = person_ds.concatenate(non_person_ds)

# Shuffle the entire dataset again. This may require too much memory.
ds["train"] = ds["train"].shuffle(2 * COUNT_PERSON_SAMPLES)
