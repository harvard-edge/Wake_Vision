import tensorflow as tf
import tensorflow_datasets as tfds

from experiment_config import cfg
import pp_ops


# A function to convert the "Train", "Validation" and "Test" parts of open images to their respective vww2 variants.
def open_images_to_vww2(ds_split, count_person_samples):
    # Use either the image level labels or bounding box labels (according to configuration) already in the open images dataset to label images as containing a person or no person
    if cfg.LABEL_TYPE == "image":
        ds_split = ds_split.map(label_person_image_labels, num_parallel_calls=tf.data.AUTOTUNE)
    elif cfg.LABEL_TYPE == "bbox":
        ds_split = ds_split.map(label_person_bbox_labels, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        raise ValueError("Configuration option \"Label Type\" must be \"image\" or \"bbox\" for the Wake Vision Dataset.")

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


def label_person_image_labels(
    ds_entry,
):
    if tf.reduce_any(
        tf.equal(tf.constant(1208, tf.int64), ds_entry["objects"]["label"])
    ):  # 1208 is the integer value for the image level label for person. Image level labels are hierarchical and thus the person label includes subclasses of persons.
        ds_entry["person"] = 1
    else:
        ds_entry["person"] = 0
    return ds_entry

def label_person_bbox_labels(
    ds_entry,
):
    # Bounding box labels include some human body parts which is hard to determine whether to label "person". We label them as -1 here so that they get selected by neither the person or the not person filter.
    if tf.reduce_any([
        tf.equal(tf.constant(68, tf.int64), ds_entry["bobjects"]["label"]), # Person 
        tf.equal(tf.constant(227, tf.int64), ds_entry["bobjects"]["label"]), # Woman
        tf.equal(tf.constant(307, tf.int64), ds_entry["bobjects"]["label"]), # Man
        tf.equal(tf.constant(332, tf.int64), ds_entry["bobjects"]["label"]), # Girl
        tf.equal(tf.constant(876, tf.int64), ds_entry["bobjects"]["label"]), # Boy
        tf.equal(tf.constant(501, tf.int64), ds_entry["bobjects"]["label"]), # Human face
        tf.equal(tf.constant(291, tf.int64), ds_entry["bobjects"]["label"]), # Human head                                              
    ]):  # Bounding box labels aren't hierarchical, and therefore we need to manually label person subclasses.
        ds_entry["person"] = 1
    elif tf.reduce_any([
        tf.equal(tf.constant(176, tf.int64), ds_entry["bobjects"]["label"]), # Human body
        tf.equal(tf.constant(14, tf.int64), ds_entry["bobjects"]["label"]), # Human eye 
        tf.equal(tf.constant(29, tf.int64), ds_entry["bobjects"]["label"]), # Skull
        tf.equal(tf.constant(147, tf.int64), ds_entry["bobjects"]["label"]), # Human mouth
        tf.equal(tf.constant(223, tf.int64), ds_entry["bobjects"]["label"]), # Human ear
        tf.equal(tf.constant(567, tf.int64), ds_entry["bobjects"]["label"]), # Human nose
        tf.equal(tf.constant(252, tf.int64), ds_entry["bobjects"]["label"]), # Human hair
        tf.equal(tf.constant(572, tf.int64), ds_entry["bobjects"]["label"]), # Human hand
        tf.equal(tf.constant(213, tf.int64), ds_entry["bobjects"]["label"]), # Human foot
        tf.equal(tf.constant(502, tf.int64), ds_entry["bobjects"]["label"]), # Human arm
        tf.equal(tf.constant(220, tf.int64), ds_entry["bobjects"]["label"]), # Human leg
        tf.equal(tf.constant(20, tf.int64), ds_entry["bobjects"]["label"]), # Beard
    ]):
        ds_entry["person"] = -1
    else:
        ds_entry["person"] = 0
    return ds_entry


def person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 1)


def non_person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 0)


def preprocessing(ds_split, batch_size=cfg.BATCH_SIZE, train=False):
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



def get_wake_vision(batch_size=cfg.BATCH_SIZE):

    ds = tfds.load(
        "open_images_v4/200k",
        data_dir=cfg.WV_DIR,
        shuffle_files=True,
    )

    ds["train"] = open_images_to_vww2(ds["train"], cfg.COUNT_PERSON_SAMPLES_TRAIN)
    ds["validation"] = open_images_to_vww2(ds["validation"], cfg.COUNT_PERSON_SAMPLES_VAL)
    ds["test"] = open_images_to_vww2(ds["test"], cfg.COUNT_PERSON_SAMPLES_TEST)

    train = preprocessing(ds["train"], batch_size, train=True)
    val = preprocessing(ds["validation"], batch_size)
    test = preprocessing(ds["test"], batch_size)

    return train, val, test
