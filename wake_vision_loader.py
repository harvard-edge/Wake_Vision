import tensorflow as tf
import tensorflow_datasets as tfds

from experiment_config import default_cfg
import pp_ops
import partial_open_images_v7.partial_open_images_v7_dataset_builder


# A function to convert the "Train", "Validation" and "Test" parts of open images to their respective vww2 variants.
def open_images_to_vww2(ds_split, count_person_samples, cfg=default_cfg):
    # Use either the image level labels or bounding box labels (according to configuration) already in the open images dataset to label images as containing a person or no person
    if cfg.LABEL_TYPE == "image":
        ds_split = ds_split.map(
            label_person_image_labels, num_parallel_calls=tf.data.AUTOTUNE
        )
    elif cfg.LABEL_TYPE == "bbox":
        
        ds_split = ds_split.map(
            lambda ds_entry: label_person_bbox_labels(ds_entry, cfg=cfg), #pass cfg to function
            num_parallel_calls=tf.data.AUTOTUNE
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
                tf.constant(14048, tf.int64), ds_entry["objects"]["label"]
            ),  # Person
            tf.equal(
                tf.constant(20610, tf.int64), ds_entry["objects"]["label"]
            ),  # Woman
            tf.equal(tf.constant(11417, tf.int64), ds_entry["objects"]["label"]),  # Man
            tf.equal(tf.constant(8000, tf.int64), ds_entry["objects"]["label"]),  # Girl
            tf.equal(tf.constant(2519, tf.int64), ds_entry["objects"]["label"]),  # Boy
            tf.equal(
                tf.constant(9270, tf.int64), ds_entry["objects"]["label"]
            ),  # Human body
            tf.equal(
                tf.constant(9274, tf.int64), ds_entry["objects"]["label"]
            ),  # Human face
            tf.equal(
                tf.constant(9279, tf.int64), ds_entry["objects"]["label"]
            ),  # Human head
            tf.equal(
                tf.constant(9266, tf.int64), ds_entry["objects"]["label"]
            ),  # Human
            tf.equal(
                tf.constant(6713, tf.int64), ds_entry["objects"]["label"]
            ),  # Female person
            tf.equal(
                tf.constant(11395, tf.int64), ds_entry["objects"]["label"]
            ),  # Male person
            tf.equal(
                tf.constant(3895, tf.int64), ds_entry["objects"]["label"]
            ),  # Child
            tf.equal(
                tf.constant(10483, tf.int64), ds_entry["objects"]["label"]
            ),  # Lady
            tf.equal(
                tf.constant(139, tf.int64), ds_entry["objects"]["label"]
            ),  # Adolescent
            tf.equal(
                tf.constant(20808, tf.int64), ds_entry["objects"]["label"]
            ),  # Youth
        ]
    ):
        ds_entry["person"] = 1
    # Image level labels include some human body parts which is hard to determine whether to label "person". We label them as -1 here so that they get selected by neither the person or the not person filter.
    elif tf.reduce_any(
        [
            tf.equal(
                tf.constant(9273, tf.int64), ds_entry["objects"]["label"]
            ),  # Human eye
            tf.equal(
                tf.constant(17150, tf.int64), ds_entry["objects"]["label"]
            ),  # Skull
            tf.equal(
                tf.constant(9282, tf.int64), ds_entry["objects"]["label"]
            ),  # Human mouth
            tf.equal(
                tf.constant(9272, tf.int64), ds_entry["objects"]["label"]
            ),  # Human ear
            tf.equal(
                tf.constant(9283, tf.int64), ds_entry["objects"]["label"]
            ),  # Human nose
            tf.equal(
                tf.constant(9276, tf.int64), ds_entry["objects"]["label"]
            ),  # Human hair
            tf.equal(
                tf.constant(9278, tf.int64), ds_entry["objects"]["label"]
            ),  # Human hand
            tf.equal(
                tf.constant(9275, tf.int64), ds_entry["objects"]["label"]
            ),  # Human foot
            tf.equal(
                tf.constant(9269, tf.int64), ds_entry["objects"]["label"]
            ),  # Human arm
            tf.equal(
                tf.constant(9281, tf.int64), ds_entry["objects"]["label"]
            ),  # Human leg
            tf.equal(
                tf.constant(1661, tf.int64), ds_entry["objects"]["label"]
            ),  # Beard
        ]
    ):
        ds_entry["person"] = -1
    else:
        ds_entry["person"] = 0
    return ds_entry


def label_person_bbox_labels(
    ds_entry, cfg=default_cfg
):
    if tf.math.equal(tf.size(ds_entry["bobjects"]["label"]), 0):
        ds_entry["person"] = -1
    elif tf.reduce_any(
        [
            check_bbox_label(ds_entry, 68, cfg=cfg),  # Person
            check_bbox_label(ds_entry, 227, cfg=cfg),  # Woman
            check_bbox_label(ds_entry, 307, cfg=cfg),  # Man
            check_bbox_label(ds_entry, 332, cfg=cfg),  # Girl
            check_bbox_label(ds_entry, 50, cfg=cfg),  # Boy
            check_bbox_label(ds_entry, 176, cfg=cfg),  # Human body
            check_bbox_label(ds_entry, 501, cfg=cfg),  # Human face
            check_bbox_label(ds_entry, 291, cfg=cfg),  # Human head
        ]
    ):
        ds_entry["person"] = 1
    # Bounding box labels include some human body parts which is hard to determine whether to label "person". We label them as -1 here so that they get selected by neither the person or the not person filter.
    elif tf.reduce_any(
        [
            tf.equal(
                tf.constant(176, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human body
            tf.equal(
                tf.constant(14, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human eye
            tf.equal(tf.constant(29, tf.int64), ds_entry["bobjects"]["label"]),  # Skull
            tf.equal(
                tf.constant(147, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human mouth
            tf.equal(
                tf.constant(223, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human ear
            tf.equal(
                tf.constant(567, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human nose
            tf.equal(
                tf.constant(252, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human hair
            tf.equal(
                tf.constant(572, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human hand
            tf.equal(
                tf.constant(213, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human foot
            tf.equal(
                tf.constant(502, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human arm
            tf.equal(
                tf.constant(220, tf.int64), ds_entry["bobjects"]["label"]
            ),  # Human leg
            tf.equal(tf.constant(20, tf.int64), ds_entry["bobjects"]["label"]),  # Beard

            #bb label is present but either too small or not in center crop
            tf.equal(tf.constant(68, tf.int64), ds_entry["bobjects"]["label"]),  # Person
            tf.equal(tf.constant(227, tf.int64), ds_entry["bobjects"]["label"]),  # Woman
            tf.equal(tf.constant(307, tf.int64), ds_entry["bobjects"]["label"]),  # Man
            tf.equal(tf.constant(332, tf.int64), ds_entry["bobjects"]["label"]),  # Girl
            tf.equal(tf.constant(50, tf.int64), ds_entry["bobjects"]["label"]),  # Boy
            tf.equal(tf.constant(501, tf.int64), ds_entry["bobjects"]["label"]),  # Human face
            tf.equal(tf.constant(291, tf.int64), ds_entry["bobjects"]["label"]),  # Human head
        ]
    ):
        ds_entry["person"] = -1
    else:
        ds_entry["person"] = 0
    return ds_entry

# This function checks for the presence of a bounding box object occupying a certain size in the ds_entry. Size can be configured in experiment_config.py.
def check_bbox_label(ds_entry, label_number, cfg=default_cfg):

    return_value = False  # This extra variable is needed as tensorflow does not allow return statements in loops.
    object_present_tensor = tf.equal(
        tf.constant(label_number, tf.int64), ds_entry["bobjects"]["label"]
    )
    bounding_boxes = ds_entry["bobjects"]["bbox"][object_present_tensor]

    #crop the bounding box area to the center crop that will happen in preprocessing.
    orig_image_h = tf.shape(ds_entry["image"])[0]
    orig_image_w = tf.shape(ds_entry["image"])[1]

    h, w = cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1]

    small_side = tf.minimum(orig_image_h, orig_image_w)
    scale =  h / small_side
    image_h = tf.cast(tf.cast(orig_image_h, tf.float64) * scale, tf.int32)
    image_w = tf.cast(tf.cast(orig_image_w, tf.float64) * scale, tf.int32)

    image_h = image_h if image_h > h else h
    image_w = image_w if image_w > w else w


    dy = (image_h - h) // 2
    dx = (image_w - w) // 2
    crop_x_min = tf.cast(dx / image_w, tf.float32)
    crop_x_max = tf.cast((dx + w) / image_w, tf.float32)
    crop_y_min = tf.cast(dy / image_h, tf.float32)
    crop_y_max = tf.cast((dy + h) / image_h, tf.float32)

    for bounding_box in bounding_boxes:
        # bbox is complete outside of crop
        if ((bounding_box[0] > crop_y_max) or 
            (bounding_box[2] < crop_y_min) or 
            (bounding_box[1] > crop_x_max) or 
            (bounding_box[3] < crop_x_min)):
            continue

        #orig pixel values of bounding box
        bb_y_min = tf.cast(bounding_box[0] * tf.cast(orig_image_h, tf.float32), tf.int32)
        bb_x_min = tf.cast(bounding_box[1] * tf.cast(orig_image_w, tf.float32), tf.int32)
        bb_y_max = tf.cast(bounding_box[2] * tf.cast(orig_image_h, tf.float32), tf.int32)
        bb_x_max = tf.cast(bounding_box[3] * tf.cast(orig_image_w, tf.float32), tf.int32)

        #rescale to new image size
        bb_y_min = tf.cast((bb_y_min - dy) / h, tf.float32)
        bb_x_min = tf.cast((bb_x_min - dx) / w, tf.float32)
        bb_y_max = tf.cast((bb_y_max - dy) / h, tf.float32)
        bb_x_max = tf.cast((bb_x_max - dx) / w, tf.float32)


        tmp_bb_y_min = bb_y_min if bounding_box[0] > crop_y_min else 0.0
        tmp_bb_y_max = bb_y_max if bounding_box[2] < crop_y_max else 1.0
        tmp_bb_x_min = bb_x_min if bounding_box[1] > crop_x_min else 0.0
        tmp_bb_x_max = bb_x_max if bounding_box[3] < crop_x_max else 1.0

        bb_effective_height = tmp_bb_y_max - tmp_bb_y_min
        bb_effective_width = tmp_bb_x_max - tmp_bb_x_min

        if (bb_effective_height * bb_effective_width) > cfg.MIN_BBOX_SIZE:
            return_value = True
    
    return return_value


def person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 1)


def non_person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 0)


def preprocessing(ds_split, batch_size, train=False, cfg=default_cfg):
    # Convert values from int8 to float32
    ds_split = ds_split.map(
        pp_ops.cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE
    )

    if train:
        # Repeat indefinitely and shuffle the dataset
        ds_split = ds_split.repeat().shuffle(cfg.SHUFFLE_BUFFER_SIZE)
        # inception crop
        ds_split = ds_split.map(
            pp_ops.inception_crop, num_parallel_calls=tf.data.AUTOTUNE
        )
        # resize
        resize = lambda ds_entry: pp_ops.resize(ds_entry, cfg.INPUT_SHAPE)
        ds_split = ds_split.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
        # flip
        ds_split = ds_split.map(
            pp_ops.random_flip_lr, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        # resize small
        resize_small = lambda ds_entry: pp_ops.resize_small(ds_entry, cfg.INPUT_SHAPE)
        ds_split = ds_split.map(
            resize_small, num_parallel_calls=tf.data.AUTOTUNE
        )
        # center crop
        center_crop = lambda ds_entry: pp_ops.center_crop(ds_entry, cfg.INPUT_SHAPE)
        ds_split = ds_split.map(center_crop, num_parallel_calls=tf.data.AUTOTUNE)

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


def get_wake_vision(cfg=default_cfg, batch_size=None):
    batch_size = batch_size or cfg.BATCH_SIZE
    ds = tfds.load(
        "partial_open_images_v7",
        data_dir=cfg.WV_DIR,
        shuffle_files=False,
    )

    ds["train"] = open_images_to_vww2(ds["train"], cfg.COUNT_PERSON_SAMPLES_TRAIN, cfg=cfg)
    ds["validation"] = open_images_to_vww2(
        ds["validation"], cfg.COUNT_PERSON_SAMPLES_VAL, cfg=cfg
    )
    ds["test"] = open_images_to_vww2(ds["test"], cfg.COUNT_PERSON_SAMPLES_TEST, cfg=cfg)

    train = preprocessing(ds["train"], batch_size, train=True, cfg=cfg)
    val = preprocessing(ds["validation"], batch_size, cfg=cfg)
    test = preprocessing(ds["test"], batch_size, cfg=cfg)

    return train, val, test


#Distance Eval
def filter_bb_area(ds_entry, min_area, max_area, cfg=default_cfg):
    
    subject_in_range = False
    addition_subject_too_close = False

    #crop the bounding box area to the center crop that will happen in preprocessing.
    orig_image_h = tf.shape(ds_entry["image"])[0]
    orig_image_w = tf.shape(ds_entry["image"])[1]

    h, w = cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1]

    small_side = tf.minimum(orig_image_h, orig_image_w)
    scale =  h / small_side
    image_h = tf.cast(tf.cast(orig_image_h, tf.float64) * scale, tf.int32)
    image_w = tf.cast(tf.cast(orig_image_w, tf.float64) * scale, tf.int32)

    image_h = image_h if image_h > h else h
    image_w = image_w if image_w > w else w


    dy = (image_h - h) // 2
    dx = (image_w - w) // 2
    crop_x_min = tf.cast(dx / image_w, tf.float32)
    crop_x_max = tf.cast((dx + w) / image_w, tf.float32)
    crop_y_min = tf.cast(dy / image_h, tf.float32)
    crop_y_max = tf.cast((dy + h) / image_h, tf.float32)
    
    for label_number in [68, 227, 307, 332, 50, 176, 501, 291]:
        object_present_tensor = tf.equal(
            tf.constant(label_number, tf.int64), ds_entry["bobjects"]["label"]
        )
        bounding_boxes = ds_entry["bobjects"]["bbox"][object_present_tensor]
        for bounding_box in bounding_boxes:
            # bbox is complete outside of crop
            if ((bounding_box[0] > crop_y_max) or 
                (bounding_box[2] < crop_y_min) or 
                (bounding_box[1] > crop_x_max) or 
                (bounding_box[3] < crop_x_min)):
                continue

            #orig pixel values of bounding box
            bb_y_min = tf.cast(bounding_box[0] * tf.cast(orig_image_h, tf.float32), tf.int32)
            bb_x_min = tf.cast(bounding_box[1] * tf.cast(orig_image_w, tf.float32), tf.int32)
            bb_y_max = tf.cast(bounding_box[2] * tf.cast(orig_image_h, tf.float32), tf.int32)
            bb_x_max = tf.cast(bounding_box[3] * tf.cast(orig_image_w, tf.float32), tf.int32)

            #rescale to new image size
            bb_y_min = tf.cast((bb_y_min - dy) / h, tf.float32)
            bb_x_min = tf.cast((bb_x_min - dx) / w, tf.float32)
            bb_y_max = tf.cast((bb_y_max - dy) / h, tf.float32)
            bb_x_max = tf.cast((bb_x_max - dx) / w, tf.float32)


            tmp_bb_y_min = bb_y_min if bounding_box[0] > crop_y_min else 0.0
            tmp_bb_y_max = bb_y_max if bounding_box[2] < crop_y_max else 1.0
            tmp_bb_x_min = bb_x_min if bounding_box[1] > crop_x_min else 0.0
            tmp_bb_x_max = bb_x_max if bounding_box[3] < crop_x_max else 1.0

            bb_effective_height = tmp_bb_y_max - tmp_bb_y_min
            bb_effective_width = tmp_bb_x_max - tmp_bb_x_min

            if (bb_effective_height * bb_effective_width) > min_area and (bb_effective_height * bb_effective_width) < max_area:
                subject_in_range = True
            elif (bb_effective_height * bb_effective_width) >= max_area:
                addition_subject_too_close = True

    return subject_in_range and not addition_subject_too_close



def get_distance_eval(cfg=default_cfg, split="test"):
    ds_test = tfds.load(
        "partial_open_images_v7",
        data_dir=cfg.WV_DIR,
        shuffle_files=False,
        split=split,
    )

    ds_test = open_images_to_vww2(ds_test, cfg.COUNT_PERSON_SAMPLES_TEST, cfg=cfg)
    no_person = ds_test.filter(lambda ds_entry: non_person_filter(ds_entry))
    far = ds_test.filter(lambda ds_entry: filter_bb_area(ds_entry, cfg.MIN_BBOX_SIZE, 0.1))#cfg.NEAR_BB_AREA))
    mid = ds_test.filter(lambda ds_entry: filter_bb_area(ds_entry, 0.1, 0.3))#cfg.MID_BB_AREA))
    near = ds_test.filter(lambda ds_entry: filter_bb_area(ds_entry, 0.3, 1.0))#cfg.FAR_BB_AREA))

    no_person = preprocessing(no_person, 1, cfg=cfg)
    far = preprocessing(far, 1, cfg=cfg)
    mid = preprocessing(mid, 1, cfg=cfg)
    near = preprocessing(near, 1, cfg=cfg)

    return {"far": far, "mid": mid, "near": near, "no_person": no_person}