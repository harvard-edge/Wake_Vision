import tensorflow as tf
from experiment_config import default_cfg


def inception_crop(ds_entry):
    """
    Inception-style crop is a random image crop (its size and aspect ratio are
    random) that was used for training Inception models, see
    https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.
    """
    image = ds_entry["image"]
    begin, crop_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        area_range=(0.6, 1.0), #TODO look into if this is too small of a min area
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    crop = tf.slice(image, begin, crop_size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])
    ds_entry["image"] = crop
    return ds_entry

def resize_small(ds_entry, input_shape=default_cfg.INPUT_SHAPE):
    #Resizes the smaller side to `smaller_size` keeping aspect ratio.
    image = ds_entry["image"]
    smaller_size = input_shape[0] # Assuming target shape is square

    h, w = tf.shape(image)[0], tf.shape(image)[1]

    # Figure out the necessary h/w.
    ratio = (
        tf.cast(smaller_size, tf.float32) /
        tf.cast(tf.minimum(h, w), tf.float32))
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)

    dtype = image.dtype
    image = tf.image.resize(image, (h, w), method="area", antialias=False)
    ds_entry["image"] = tf.cast(image, dtype)
    return ds_entry

def center_crop(ds_entry, input_shape=default_cfg.INPUT_SHAPE):
    #crop image to desired size
    image = ds_entry["image"]
    h, w = input_shape[0], input_shape[1]
    dy = (tf.shape(image)[0] - h) // 2
    dx = (tf.shape(image)[1] - w) // 2
    ds_entry["image"] = tf.image.crop_to_bounding_box(image, dy, dx, h, w)
    return ds_entry


def resize(ds_entry, input_shape=default_cfg.INPUT_SHAPE):
    ds_entry["image"] = tf.image.resize(ds_entry["image"], input_shape[:2])
    return ds_entry


def cast_images_to_float32(ds_entry):
    ds_entry["image"] = tf.cast(ds_entry["image"], tf.float32)
    return ds_entry


def mobilenet_preprocessing_wrapper(ds_entry):
    ds_entry["image"] = tf.keras.applications.mobilenet_v2.preprocess_input(
        ds_entry["image"]
    )
    return ds_entry


def prepare_supervised(ds_entry):
    return (ds_entry["image"], ds_entry["person"])


def vww_rename(ds_entry):
    ds_entry["image"] = ds_entry["image/encoded"]
    ds_entry["person"] = ds_entry["image/class/label"]
    return ds_entry


def random_flip_lr(ds_entry):
    ds_entry["image"] = tf.image.random_flip_left_right(ds_entry["image"])
    return ds_entry