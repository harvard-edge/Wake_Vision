import tensorflow as tf
from experiment_config import default_cfg


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

            if (bb_effective_height * bb_effective_width) > min_area and (
                bb_effective_height * bb_effective_width
            ) < max_area:
                subject_in_range = True
            elif (bb_effective_height * bb_effective_width) >= max_area:
                addition_subject_too_close = True

    return subject_in_range and not addition_subject_too_close
