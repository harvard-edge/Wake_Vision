# This file contains functions to create the wake vision dataset from open images v4
# At the current time it expects to be imported into another script that calls its "create_wake_vision_ds" function.
# In the future we may support saving the dataset to persistent memory

import tensorflow as tf
import tensorflow_datasets as tfds

# Config
DATA_DIR = "./dataset/"
COUNT_PERSON_SAMPLES_TRAIN = 844965  # Number of person samples in the train sdataset. The number of non-person samples are 898077. We will use this number to balance the dataset.
COUNT_PERSON_SAMPLES_VAL = 9973  # There are 31647 non-person samples.
COUNT_PERSON_SAMPLES_TEST = 30226  # There are 95210 non-person samples. The distribution of persons in both the Val and Test set is close to 24% (Val:23.96) (Test:24.09) so we may not need to reduce the size of these.


# The "main" function used to create the wake vision dataset
def create_wake_vision_ds():
    ds = tfds.load(
        "open_images_v4/200k",
        data_dir=DATA_DIR,
        shuffle_files=True,
    )
    ds["train"] = open_images_to_vww2(ds["train"], COUNT_PERSON_SAMPLES_TRAIN)
    ds["validation"] = open_images_to_vww2(ds["validation"], COUNT_PERSON_SAMPLES_VAL)
    ds["test"] = open_images_to_vww2(ds["test"], COUNT_PERSON_SAMPLES_TEST)
    return ds


# A function to convert the "Train", "Validation" and "Test" parts of open images to their respective vww2 variants.
def open_images_to_vww2(ds_split, count_person_samples):
    # Use the image level classes already in the open images dataset to label images as containing a person or no person
    ds_split = ds_split.map(label_person, num_parallel_calls=tf.data.AUTOTUNE)

    # Filter the dataset into a part with persons and a part with no persons
    person_ds = ds_split.filter(person_filter)
    non_person_ds = ds_split.filter(non_person_filter)

    # Take an equal amount of images with persons and with no persons.
    person_ds = person_ds.take(count_person_samples)
    non_person_ds = non_person_ds.take(count_person_samples)

    # We now interleave these two datasets with an equal probability of picking an element from each dataset. This should result in a shuffled dataset.
    # As an added benefit this allows us to shuffle the dataset differently for every epoch using "rerandomize_each_iteration".
    ds_split = tf.data.Dataset.sample_from_datasets(
        [person_ds, non_person_ds],
        stop_on_empty_dataset=False,
        rerandomize_each_iteration=True,
    )

    return ds_split


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
