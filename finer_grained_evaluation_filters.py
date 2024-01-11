import tensorflow as tf

# MIAP Filters


def get_miap_set(ds, miap_subset: str, miap_subset_category: str):
    return ds.filter(
        lambda x: tf.reduce_any(tf.equal(x["miaps"][miap_subset], miap_subset_category))
        and tf.math.not_equal(tf.size(x["miaps"][miap_subset]), 0)
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
def get_image_lighting(image):
    # First convert the image to greyscale
    greyscale_image = tf.image.rgb_to_grayscale(image)

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
