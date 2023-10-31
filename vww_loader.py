import tensorflow as tf
import tensorflow_datasets as tfds


import experiment_config as cfg


def preprocessing(ds_split, batch_size=cfg.BATCH_SIZE):
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
    return ds_split.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def resize_images(ds_entry):
    ds_entry["image"] = tf.keras.preprocessing.image.smart_resize(
        ds_entry["image/encoded"], cfg.INPUT_SHAPE[:2], interpolation="bilinear"
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
    return (ds_entry["image"], ds_entry["image/class/label"])


def get_vww(batch_size=cfg.BATCH_SIZE):

    builder = tfds.builder_from_directory(cfg.VWW_DIR)
    ds = builder.as_dataset()

    train = preprocessing(ds["train"], batch_size)
    val = train
    train = train
    test = preprocessing(ds["val"], batch_size)

    return train, val, test
