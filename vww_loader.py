import tensorflow as tf
import tensorflow_datasets as tfds


import experiment_config as cfg
import pp_ops


def preprocessing(ds_split, batch_size=cfg.BATCH_SIZE, train=False):
    #vww has irreglar names for image and label
    ds_split = ds_split.map(pp_ops.vww_rename, num_parallel_calls=tf.data.AUTOTUNE)

    # Convert values from int8 to float32
    ds_split = ds_split.map(pp_ops.cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE)

    if train:
        # inception crop
        ds_split = ds_split.map(pp_ops.inception_crop, num_parallel_calls=tf.data.AUTOTUNE)
        # resize
        ds_split = ds_split.map(pp_ops.resize, num_parallel_calls=tf.data.AUTOTUNE)
        # flip
        ds_split = ds_split.map(pp_ops.random_flip_lr, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # resize small
        ds_split = ds_split.map(pp_ops.resize_small, num_parallel_calls=tf.data.AUTOTUNE)
        # center crop
        ds_split = ds_split.map(pp_ops.center_crop, num_parallel_calls=tf.data.AUTOTUNE)

    # Use the official mobilenet preprocessing to normalize images
    ds_split = ds_split.map(
        pp_ops.mobilenet_preprocessing_wrapper, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert each dataset entry from a dictionary to a tuple of (img, label) to be used by the keras API.
    ds_split = ds_split.map(pp_ops.prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch the dataset for improved performance
    return ds_split.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_vww(batch_size=cfg.BATCH_SIZE):
    builder = tfds.builder_from_directory(cfg.VWW_DIR)

    train = preprocessing(builder.as_dataset(split="train[:90%]"), batch_size, train=True)
    val = preprocessing(builder.as_dataset(split="train[90%:]"), batch_size)
    test = preprocessing(builder.as_dataset(split="val"), batch_size)

    return train, val, test
