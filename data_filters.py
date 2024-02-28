import tensorflow as tf
from experiment_config import default_cfg


# This function checks for the presence of an image level label with at least MIN_IMAGE_LEVEL_CONFIDENCE confidence in the ds_entry.
def check_image_level_label(ds_entry, label_number, cfg=default_cfg):
    object_present_tensor = tf.equal(
        tf.constant(label_number, tf.int64), ds_entry["objects"]["label"]
    )
    confidence = ds_entry["objects"]["confidence"][object_present_tensor]

    if tf.size(confidence) == 0:
        return False

    confident_object_present_tensor = tf.math.greater_equal(
        confidence, cfg.MIN_IMAGE_LEVEL_CONFIDENCE
    )

    # If any of the image level labels with label_number are present with a confidence greater than MIN_IMAGE_LEVEL_CONFIDENCE then return True.
    return_value = tf.reduce_any(confident_object_present_tensor)

    return return_value

# This function checks for the presence of a bounding box object occupying a certain size in the ds_entry. Size can be configured in experiment_config.py.
def check_bbox_label(
    ds_entry,
    label_number,
    cfg,
    exclude_depiction=True,
    exclude_outside_crop=True,
):
    object_present_tensor = tf.equal(
        tf.constant(label_number, tf.int64), ds_entry["bobjects"]["label"]
    )

    if exclude_depiction:
        # Remove the positive values from object_present_tensor that stem from depictions.
        non_depiction_tensor = tf.equal(
            tf.constant(0, tf.int8), ds_entry["bobjects"]["is_depiction"]
        )
        object_present_tensor = tf.logical_and(object_present_tensor, non_depiction_tensor)

    if tf.logical_not(tf.reduce_any(object_present_tensor)):
        return False
    elif tf.logical_not(exclude_outside_crop):
        return True
    else:
        bounding_boxes = ds_entry["bobjects"]["bbox"][object_present_tensor]
        return check_bbox_inside_crop(ds_entry, bounding_boxes, cfg)


def check_bbox_inside_crop(ds_entry, bounding_boxes, cfg):
    return_value = False
    # crop the bounding box area to the center crop that will happen in preprocessing.
    orig_image_h = tf.shape(ds_entry["image"])[0]
    orig_image_w = tf.shape(ds_entry["image"])[1]

    h, w = cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1]

    small_side = tf.minimum(orig_image_h, orig_image_w)
    scale = h / small_side
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
        if (
            (bounding_box[0] > crop_y_max)
            or (bounding_box[2] < crop_y_min)
            or (bounding_box[1] > crop_x_max)
            or (bounding_box[3] < crop_x_min)
        ):
            continue

        # orig pixel values of bounding box
        bb_y_min = tf.cast(
            bounding_box[0] * tf.cast(orig_image_h, tf.float32), tf.int32
        )
        bb_x_min = tf.cast(
            bounding_box[1] * tf.cast(orig_image_w, tf.float32), tf.int32
        )
        bb_y_max = tf.cast(
            bounding_box[2] * tf.cast(orig_image_h, tf.float32), tf.int32
        )
        bb_x_max = tf.cast(
            bounding_box[3] * tf.cast(orig_image_w, tf.float32), tf.int32
        )

        # rescale to new image size
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

# MIAP Filters
def get_miap_set(ds, miap_subset: str, miap_subset_category: str):
    return ds.filter(
        lambda x: tf.reduce_all(
            tf.equal(x["miaps"][miap_subset], miap_subset_category)
        )  # Check the desired MIAP label is present in all MIAP boxes
        and tf.math.not_equal(
            tf.size(x["miaps"][miap_subset]), 0
        )  # Check that MIAP labels exist for this image
        and tf.equal(
            x["person"], 1
        )  # Check that there is a person in the centre crop of the image
    )


def get_predominantly_female_set(ds):
    return get_miap_set(
        ds, "gender_presentation", tf.constant(0, tf.int64)
    )  # Predominantly Feminine


def get_predominantly_male_set(ds):
    return get_miap_set(
        ds, "gender_presentation", tf.constant(1, tf.int64)
    )  # Predominantly Masculine


def get_unknown_gender_set(ds):
    return get_miap_set(ds, "gender_presentation", tf.constant(2, tf.int64))  # Unknown


def get_young_set(ds):
    return get_miap_set(ds, "age_presentation", tf.constant(0, tf.int64))  # Young


def get_middle_set(ds):
    return get_miap_set(ds, "age_presentation", tf.constant(1, tf.int64))  # Middle


def get_older_set(ds):
    return get_miap_set(ds, "age_presentation", tf.constant(2, tf.int64))  # Older


def get_unknown_age_set(ds):
    return get_miap_set(ds, "age_presentation", tf.constant(3, tf.int64))  # Unknown


# Lighting filters
def get_image_lighting(ds_sample):
    # First convert the image to greyscale
    greyscale_image = tf.image.rgb_to_grayscale(ds_sample["image"])

    # Return the average pixel value of the new greyscale image
    return tf.reduce_mean(greyscale_image)


def get_low_lighting(ds):
    return ds.filter(lambda image: get_image_lighting(image) < 85)


def get_medium_lighting(ds):
    return ds.filter(
        lambda image: get_image_lighting(image) >= 85
        and get_image_lighting(image) < 170
    )


def get_high_lighting(ds):
    return ds.filter(lambda image: get_image_lighting(image) >= 170)


def filter_bb_area(ds_entry, min_area, max_area, cfg=default_cfg):
    subject_in_range = False
    addition_subject_too_close = False

    # crop the bounding box area to the center crop that will happen in preprocessing.
    orig_image_h = tf.shape(ds_entry["image"])[0]
    orig_image_w = tf.shape(ds_entry["image"])[1]

    h, w = cfg.INPUT_SHAPE[0], cfg.INPUT_SHAPE[1]

    small_side = tf.minimum(orig_image_h, orig_image_w)
    scale = h / small_side
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
            if (
                (bounding_box[0] > crop_y_max)
                or (bounding_box[2] < crop_y_min)
                or (bounding_box[1] > crop_x_max)
                or (bounding_box[3] < crop_x_min)
            ):
                continue

            # orig pixel values of bounding box
            bb_y_min = tf.cast(
                bounding_box[0] * tf.cast(orig_image_h, tf.float32), tf.int32
            )
            bb_x_min = tf.cast(
                bounding_box[1] * tf.cast(orig_image_w, tf.float32), tf.int32
            )
            bb_y_max = tf.cast(
                bounding_box[2] * tf.cast(orig_image_h, tf.float32), tf.int32
            )
            bb_x_max = tf.cast(
                bounding_box[3] * tf.cast(orig_image_w, tf.float32), tf.int32
            )

            # rescale to new image size
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

            if tf.logical_and(
                tf.math.greater((bb_effective_height * bb_effective_width), min_area),
                tf.math.less((bb_effective_height * bb_effective_width), max_area)
            ):
                subject_in_range = True
            elif tf.math.greater_equal((bb_effective_height * bb_effective_width), max_area):
                addition_subject_too_close = True

    return subject_in_range and not addition_subject_too_close

def body_part_filter(ds_entry, body_part_label_num, other_body_parts, cfg=default_cfg):
    # Check that the image contains the specified body part
    # in the center crop.
    target_body_part_present = check_bbox_label(ds_entry, body_part_label_num, cfg=cfg)
    
    #check that no other body parts are present anywhere in the image
    other_body_parts_present = tf.reduce_any(list(
        check_bbox_label(ds_entry, body_part_label, cfg=cfg, exclude_outside_crop=False)
            for body_part_label in other_body_parts
    ))
    return target_body_part_present and not other_body_parts_present


def get_body_part_set(ds, body_part_label_num, cfg=default_cfg):
    other_body_parts = cfg.BBOX_BODY_PART_DICTIONARY.values()
    other_body_parts.remove(body_part_label_num)
    if body_part_label_num == cfg.BBOX_BODY_PART_DICTIONARY["Human foot"]:
        other_body_parts.remove(cfg.BBOX_BODY_PART_DICTIONARY["Human leg"])
    
    ds = ds.filter(person_filter) #ensures the image contains a person
    
    # Filter the dataset to only include images with the specified body part
    # and no other body parts in the image
    return ds.filter(lambda ds_entry: body_part_filter(ds_entry, body_part_label_num, other_body_parts, cfg=cfg))


def depiction_eval_filter(ds_entry, return_all_depictions=False, return_person_depictions=True):
    depiction_tensor = tf.equal(
            tf.constant(1, tf.int8), ds_entry["bobjects"]["is_depiction"]
        ) # Check if the image contains a depiction
    if tf.logical_not(tf.reduce_any(depiction_tensor)):
        return False
    
    if return_all_depictions:
        return True
    
    # Check if the depiction is of a person
    depiction_is_person = False
    for label_number in [68, 227, 307, 332, 50, 176, 501, 291]:
        object_present_tensor = tf.equal(
            tf.constant(label_number, tf.int64), ds_entry["bobjects"]["label"]
        )
        if tf.reduce_any(tf.logical_and(object_present_tensor, depiction_tensor)):
            depiction_is_person = True
    
    # Return the depictions of people if return_person_depictions is True
    # or the depictions of non-people if return_person_depictions is False
    return depiction_is_person == return_person_depictions