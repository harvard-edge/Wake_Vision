# This file contains functions to prepare the wake vision dataset for the format expected by mobilenet_v1

BATCH_SIZE = 96
TARGET_IMAGE_RESOLUTION = (
    224,
    224,
)

import tensorflow as tf


def mobilenetv1_preprocessing(ds_split, one_hot=False):
    # Crop images to the resolution expected by mobilenet
    ds_split = ds_split.map(_resize_images, num_parallel_calls=tf.data.AUTOTUNE)

    # Convert values from int8 to float32
    ds_split = ds_split.map(
        _cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Use the official mobilenet preprocessing to normalize images
    ds_split = ds_split.map(
        _mobilenet_preprocessing_wrapper, num_parallel_calls=tf.data.AUTOTUNE
    )

    # If a one hot encoding is desired then change the labels to be one hot encoded
    if one_hot:
        ds_split = ds_split.map(
            _sparse_labels_to_one_hot, num_parallel_calls=tf.data.AUTOTUNE
        )

    # Convert each dataset entry from a dictionary to a tuple of (img, label) to be used by the keras API.
    ds_split = ds_split.map(_prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE)

    # Prefetch the dataset for improved performance
    return ds_split.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def _resize_images(ds_entry):
    ds_entry["image"] = tf.keras.preprocessing.image.smart_resize(
        ds_entry["image"], TARGET_IMAGE_RESOLUTION, interpolation="bilinear"
    )
    return ds_entry


def _cast_images_to_float32(ds_entry):
    ds_entry["image"] = tf.cast(ds_entry["image"], tf.float32)
    return ds_entry


def _mobilenet_preprocessing_wrapper(ds_entry):
    ds_entry["image"] = tf.keras.applications.mobilenet.preprocess_input(
        ds_entry["image"]
    )
    return ds_entry


def _sparse_labels_to_one_hot(ds_entry):
    ds_entry["person"] = tf.one_hot(ds_entry["person"], 2)
    return ds_entry


def _prepare_supervised(ds_entry):
    return (ds_entry["image"], ds_entry["person"])
