# This file contains functions to prepare the wake vision dataset for the format expected by mobilenet_v1

import tensorflow as tf


def mobilenetv1_preprocessing(ds_split):
    # Crop images to the resolution expected by mobilenet
    ds_split = ds_split.map(resize_images, num_parallel_calls=tf.data.AUTOTUNE)

    # Convert values from int8 to float32
    ds_split = ds_split.map(cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE)

    # Use the official mobilenet preprocessing to normalize images
    ds_split = ds_split.map(
        mobilenet_preprocessing_wrapper, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert each dataset entry from a dictionary to a tuple of (img, label) to be used by the keras API.
    ds_split = ds_split.map(prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch the dataset for improved performance
    return ds_split.batch(96).prefetch(tf.data.AUTOTUNE)


def resize_images(ds_entry):
    ds_entry["image"] = tf.keras.preprocessing.image.smart_resize(
        ds_entry["image"], (96, 96), interpolation="bilinear"
    )
    return ds_entry


def cast_images_to_float32(ds_entry):
    ds_entry["image"] = tf.cast(ds_entry["image"], tf.float32)
    return ds_entry


def mobilenet_preprocessing_wrapper(ds_entry):
    ds_entry["image"] = tf.keras.applications.mobilenet.preprocess_input(
        ds_entry["image"]
    )
    return ds_entry


def prepare_supervised(ds_entry):
    return (ds_entry["image"], ds_entry["person"])
