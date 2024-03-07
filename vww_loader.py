import tensorflow as tf
import tensorflow_datasets as tfds

from experiment_config import default_cfg
import pp_ops


def preprocessing(ds_split, batch_size, train=False, cfg=default_cfg):
    #vww has irreglar names for image and label
    ds_split = ds_split.map(pp_ops.vww_rename, num_parallel_calls=tf.data.AUTOTUNE)

    # Convert values from int8 to float32
    ds_split = ds_split.map(pp_ops.cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE)

    if train:
        # Repeat indefinitely and shuffle the dataset
        ds_split = ds_split.repeat().shuffle(cfg.SHUFFLE_BUFFER_SIZE)
        # inception crop
        ds_split = ds_split.map(pp_ops.inception_crop, num_parallel_calls=tf.data.AUTOTUNE)
        # resize
        resize = lambda ds_entry: pp_ops.resize(ds_entry, cfg.INPUT_SHAPE)
        ds_split = ds_split.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
        # flip
        ds_split = ds_split.map(pp_ops.random_flip_lr, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # resize small
        resize_small = lambda ds_entry: pp_ops.resize_small(ds_entry, cfg.INPUT_SHAPE)
        ds_split = ds_split.map(resize_small, num_parallel_calls=tf.data.AUTOTUNE)
        # center crop
        center_crop = lambda ds_entry: pp_ops.center_crop(ds_entry, cfg.INPUT_SHAPE)
        ds_split = ds_split.map(center_crop, num_parallel_calls=tf.data.AUTOTUNE)
        
    if cfg.grayscale:
        ds_split = ds_split.map(
            pp_ops.grayscale, num_parallel_calls=tf.data.AUTOTUNE
        )

    # Use the official mobilenet preprocessing to normalize images
    ds_split = ds_split.map(
        pp_ops.mobilenet_preprocessing_wrapper, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert each dataset entry from a dictionary to a tuple of (img, label) to be used by the keras API.
    ds_split = ds_split.map(pp_ops.prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch the dataset for improved performance
    return ds_split.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_vww(cfg=default_cfg, batch_size=None):
    batch_size = batch_size or cfg.BATCH_SIZE
    builder = tfds.builder_from_directory(cfg.VWW_DIR)

    train = preprocessing(builder.as_dataset(split="train[:90%]"), batch_size, train=True, cfg=cfg)
    val = preprocessing(builder.as_dataset(split="train[90%:]"), batch_size, cfg=cfg)
    test = preprocessing(builder.as_dataset(split="val"), batch_size, cfg=cfg)

    return train, val, test
