import tensorflow as tf
import tensorflow_datasets as tfds

# Config
DATA_DIR = "./dataset/"
COUNT_PERSON_SAMPLES = 844965  # Number of person samples in the dataset. The number of non-person samples are 898077. We will use this number to balance the dataset.

ds = tfds.load(
    "open_images_v4/200k",
    data_dir=DATA_DIR,
    shuffle_files=True,
)


def label_person(
    ds_entry,
):
    if tf.reduce_any(
        tf.equal(tf.constant(1208, tf.int64), ds_entry["objects"]["label"])
    ):  # 1208 is the integer value for the image level label for person
        ds_entry["person"] = 1
    else:
        ds_entry["person"] = 0
    return ds_entry


def person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 1)


def non_person_filter(ds_entry):
    return tf.equal(ds_entry["person"], 0)


def crop_images(ds_entry):
    ds_entry["image"] = tf.image.resize_with_crop_or_pad(ds_entry["image"], 224, 224)
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
    return (ds_entry["image"], ds_entry["person"])


ds["train"] = ds["train"].map(label_person, num_parallel_calls=tf.data.AUTOTUNE)

person_ds = ds["train"].filter(person_filter)
non_person_ds = ds["train"].filter(non_person_filter)

person_ds = person_ds.take(COUNT_PERSON_SAMPLES)
non_person_ds = non_person_ds.take(COUNT_PERSON_SAMPLES)

# We now interleave these two datasets with an equal probability of picking an element from each dataset. This should result in a shuffled dataset.
# As an added benefit this allows us to shuffle the dataset differently for every epoch using "rerandomize_each_iteration".
ds["train"] = tf.data.Dataset.sample_from_datasets(
    [person_ds, non_person_ds],
    stop_on_empty_dataset=False,
    rerandomize_each_iteration=True,
)


## Start of inference testing part
# Try out training and inference with mobilenet v1
# Parameters given to models are the same as for the models used in the visual wake words paper

# Optimizer
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.045, momentum=0.9, weight_decay=0.9
)

# Loss
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# Crop images to the resolution expected by mobilenet
mobilenetv1_train = ds["train"].map(crop_images, num_parallel_calls=tf.data.AUTOTUNE)

# Convert values from int8 to float32
mobilenetv1_train = mobilenetv1_train.map(
    cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE
)

# Use the official mobilenet preprocessing to normalize images
mobilenetv1_train = mobilenetv1_train.map(
    mobilenet_preprocessing_wrapper, num_parallel_calls=tf.data.AUTOTUNE
)

# Convert each dataset entry from a dictionary to a tuple of (img, label) to be used by the keras API.
mobilenetv1_train = mobilenetv1_train.map(
    prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE
)

mobilenetv1_train = mobilenetv1_train.batch(96).prefetch(tf.data.AUTOTUNE)

# The visual wake words paper mention that the depth multiplier of their mobilenet model is 0.25.
# This is however not a possible value for the depth multiplier parameter of this api. There may be some termonology problems here where what the paper calls depth multiplier is the alpha parameter of the api.
mobilenetv1 = tf.keras.applications.MobileNet(alpha=0.25, weights=None, classes=2)

mobilenetv1.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

mobilenetv1.fit(mobilenetv1_train, epochs=10)
