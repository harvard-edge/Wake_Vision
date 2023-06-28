import numpy as np
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
    return ds_entry


def person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 1)


def non_person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 0)


def count_ds(ds_entry, _):
    return ds_entry + 1


ds["train"] = ds["train"].map(label_person, num_parallel_calls=tf.data.AUTOTUNE)

person_ds = ds["train"].filter(person_filter)
non_person_ds = ds["train"].filter(non_person_filter)

count_person_ds = person_ds.reduce(np.int64(0), count_ds)  # Should be 844965
count_non_person_ds = non_person_ds.reduce(np.int64(0), count_ds)  # Should be 898077

if count_person_ds < count_non_person_ds:
    non_person_ds = non_person_ds.take(count_person_ds)
else:
    person_ds = person_ds.take(count_non_person_ds)
