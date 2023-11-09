import tensorflow as tf
import tensorflow_datasets as tfds

from experiment_config import cfg
import pp_ops


# A function to convert the "Train", "Validation" and "Test" parts of open images to their respective vww2 variants.
def open_images_to_vww2(ds_split, count_person_samples):
    # Use either the image level labels or bounding box labels (according to configuration) already in the open images dataset to label images as containing a person or no person
    if cfg.LABEL_TYPE == "image":
        ds_split = ds_split.map(
            label_person_image_labels, num_parallel_calls=tf.data.AUTOTUNE
        )
    elif cfg.LABEL_TYPE == "bbox":
        ds_split = ds_split.map(
            label_person_bbox_labels, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        raise ValueError(
            'Configuration option "Label Type" must be "image" or "bbox" for the Wake Vision Dataset.'
        )

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
        [
            tf.equal(
                tf.constant(1208, tf.int64), ds_entry["objects"]["label"]
            ),  # Person
            tf.equal(
                tf.constant(7212, tf.int64), ds_entry["objects"]["label"]
            ),  # Woman
            tf.equal(tf.constant(10693, tf.int64), ds_entry["objects"]["label"]),  # Man
            tf.equal(
                tf.constant(11877, tf.int64), ds_entry["objects"]["label"]
            ),  # Girl
            tf.equal(tf.constant(876, tf.int64), ds_entry["objects"]["label"]),  # Boy
            tf.equal(
                tf.constant(17410, tf.int64), ds_entry["objects"]["label"]
            ),  # Human face
            tf.equal(
                tf.constant(9930, tf.int64), ds_entry["objects"]["label"]
            ),  # Human head
            tf.equal(
                tf.constant(17112, tf.int64), ds_entry["objects"]["label"]
            ),  # Human
            tf.equal(
                tf.constant(6637, tf.int64), ds_entry["objects"]["label"]
            ),  # Female person
            tf.equal(
                tf.constant(12187, tf.int64), ds_entry["objects"]["label"]
            ),  # Male person
            tf.equal(
                tf.constant(19977, tf.int64), ds_entry["objects"]["label"]
            ),  # Child
            tf.equal(tf.constant(5201, tf.int64), ds_entry["objects"]["label"]),  # Lady
            tf.equal(
                tf.constant(19617, tf.int64), ds_entry["objects"]["label"]
            ),  # Adolescent
            tf.equal(
                tf.constant(2873, tf.int64), ds_entry["objects"]["label"]
            ),  # Youth
        ]
    ):
        ds_entry["person"] = 1
    # Image level labels include some human body parts which is hard to determine whether to label "person". We label them as -1 here so that they get selected by neither the person or the not person filter.
    elif tf.reduce_any(
        [
            tf.equal(
                tf.constant(5075, tf.int64), ds_entry["objects"]["label"]
            ),  # Human body
            tf.equal(
                tf.constant(311, tf.int64), ds_entry["objects"]["label"]
            ),  # Human eye
            tf.equal(tf.constant(483, tf.int64), ds_entry["objects"]["label"]),  # Skull
            tf.equal(
                tf.constant(4129, tf.int64), ds_entry["objects"]["label"]
            ),  # Human mouth
            tf.equal(
                tf.constant(7172, tf.int64), ds_entry["objects"]["label"]
            ),  # Human ear
            tf.equal(
                tf.constant(19450, tf.int64), ds_entry["objects"]["label"]
            ),  # Human nose
            tf.equal(
                tf.constant(8309, tf.int64), ds_entry["objects"]["label"]
            ),  # Human hair
            tf.equal(
                tf.constant(19486, tf.int64), ds_entry["objects"]["label"]
            ),  # Human hand
            tf.equal(
                tf.constant(6750, tf.int64), ds_entry["objects"]["label"]
            ),  # Human foot
            tf.equal(
                tf.constant(17415, tf.int64), ds_entry["objects"]["label"]
            ),  # Human arm
            tf.equal(
                tf.constant(6966, tf.int64), ds_entry["objects"]["label"]
            ),  # Human leg
            tf.equal(tf.constant(369, tf.int64), ds_entry["objects"]["label"]),  # Beard
        ]
    ):
        ds_entry["person"] = -1
    else:
        ds_entry["person"] = 0
    return ds_entry


def label_person_bbox_labels(
    ds_entry,
):
    if tf.reduce_any(
        [
            check_bbox_label(ds_entry, 68),  # Person
            check_bbox_label(ds_entry, 227),  # Woman
            check_bbox_label(ds_entry, 307),  # Man
            check_bbox_label(ds_entry, 332),  # Girl
            check_bbox_label(ds_entry, 50),  # Boy
            check_bbox_label(ds_entry, 501),  # Human face
            check_bbox_label(ds_entry, 291),  # Human head
        ]
    ):
        ds_entry["person"] = 1
    # Bounding box labels include some human body parts which is hard to determine whether to label "person". We label them as -1 here so that they get selected by neither the person or the not person filter.
    elif tf.reduce_any(
        [
            check_bbox_label(ds_entry, 176),  # Human body
            check_bbox_label(ds_entry, 14),  # Human eye
            check_bbox_label(ds_entry, 29),  # Skull
            check_bbox_label(ds_entry, 147),  # Human mouth
            check_bbox_label(ds_entry, 223),  # Human ear
            check_bbox_label(ds_entry, 567),  # Human nose
            check_bbox_label(ds_entry, 252),  # Human hair
            check_bbox_label(ds_entry, 572),  # Human hand
            check_bbox_label(ds_entry, 213),  # Human foot
            check_bbox_label(ds_entry, 502),  # Human arm
            check_bbox_label(ds_entry, 220),  # Human leg
            check_bbox_label(ds_entry, 20),  # Beard
        ]
    ):
        ds_entry["person"] = -1
    else:
        ds_entry["person"] = 0
    return ds_entry


# This function checks for the presence of a bounding box object occupying a certain size in the ds_entry. Size can be configured in experiment_config.py.
def check_bbox_label(ds_entry, label_number):
    return_value = False  # This extra variable is needed as tensorflow does not allow return statements in loops.
    object_present_tensor = tf.equal(
        tf.constant(label_number, tf.int64), ds_entry["bobjects"]["label"]
    )
    bounding_boxes = ds_entry["bobjects"]["bbox"][object_present_tensor]
    for bounding_box in bounding_boxes:
        if (
            (bounding_box[2] - bounding_box[0]) * (bounding_box[3] - bounding_box[1])
        ) > cfg.MIN_BBOX_SIZE:
            return_value = True
    return return_value


def person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 1)


def non_person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 0)


def preprocessing(ds_split, batch_size=cfg.BATCH_SIZE, train=False):
    # Convert values from int8 to float32
    ds_split = ds_split.map(
        pp_ops.cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE
    )

    if train:
        # inception crop
        ds_split = ds_split.map(
            pp_ops.inception_crop, num_parallel_calls=tf.data.AUTOTUNE
        )
        # resize
        ds_split = ds_split.map(pp_ops.resize, num_parallel_calls=tf.data.AUTOTUNE)
        # flip
        ds_split = ds_split.map(
            pp_ops.random_flip_lr, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        # resize small
        ds_split = ds_split.map(
            pp_ops.resize_small, num_parallel_calls=tf.data.AUTOTUNE
        )
        # center crop
        ds_split = ds_split.map(pp_ops.center_crop, num_parallel_calls=tf.data.AUTOTUNE)

    # Use the official mobilenet preprocessing to normalize images
    ds_split = ds_split.map(
        pp_ops.mobilenet_preprocessing_wrapper, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert each dataset entry from a dictionary to a tuple of (img, label) to be used by the keras API.
    ds_split = ds_split.map(
        pp_ops.prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch the dataset for improved performance
    return ds_split.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def get_wake_vision(batch_size=cfg.BATCH_SIZE):
    ds = tfds.load(
        "open_images_v4/200k",
        data_dir=cfg.WV_DIR,
        shuffle_files=True,
    )

    ds["train"] = open_images_to_vww2(ds["train"], cfg.COUNT_PERSON_SAMPLES_TRAIN)
    ds["validation"] = open_images_to_vww2(
        ds["validation"], cfg.COUNT_PERSON_SAMPLES_VAL
    )
    ds["test"] = open_images_to_vww2(ds["test"], cfg.COUNT_PERSON_SAMPLES_TEST)

    train = preprocessing(ds["train"], batch_size, train=True)
    val = preprocessing(ds["validation"], batch_size)
    test = preprocessing(ds["test"], batch_size)

    return train, val, test
