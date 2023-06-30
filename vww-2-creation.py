import numpy as np
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
    return tf.image.resize_with_crop_or_pad(ds_entry, 224, 224)


ds["train"] = ds["train"].map(label_person, num_parallel_calls=tf.data.AUTOTUNE)

person_ds = ds["train"].filter(person_filter)
non_person_ds = ds["train"].filter(non_person_filter)

non_person_ds = non_person_ds.take(COUNT_PERSON_SAMPLES)
person_ds = person_ds.take(COUNT_PERSON_SAMPLES)

ds["train"] = person_ds.concatenate(non_person_ds)

# Shuffle the entire dataset again. This may require too much memory.
ds["train"] = ds["train"].shuffle(2 * COUNT_PERSON_SAMPLES)

## Start of inference testing part

# Optimizer
optimizer = tf.keras.optimizers.RMSprop(
    learning_rate=0.045, momentum=0.9, weight_decay=0.9
)

# Loss
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# Decay & Momentum: 0.9
# Batch size: 96
# Learning rate: 0.045 initially decaying by .98 per epoch.

# Conv batch norm decay: 0.99
# Trained with quantization aware training 10^-5 learning rate and 0.9 decay

# Try out training and inference with mobilenet v1
# Parameters given to models are the same as for the models used in the visual wake words paper

mobilenetv1_train = ds["train"].map(crop_images, num_parallel_calls=tf.data.AUTOTUNE)
mobilenetv1_train = tf.keras.applications.mobilenet.preprocess_input(mobilenetv1_train)

mobilenetv1_train = mobilenetv1_train.batch(96).prefetch(tf.data.AUTOTUNE)

mobilenetv1 = tf.keras.applications.MobileNet(
    depth_multiplier=0.25, weights=None, classes=2
)

mobilenetv1.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

mobilenetv1.fit(ds["train"]["image"], ds["train"]["person"], epochs=10)

# Also try out Mobilenet v2
# mobilenetv2_train = ds["train"].map(crop_images, num_parallel_calls=tf.data.AUTOTUNE)
# mobilenetv2_train = tf.keras.applications.mobilenet_v2.preprocess_input(
#    mobilenetv2_train
# )
#
## The depth multiplier should be set to 0.5 but this does not seem to be supported in this api.
# mobilenetv2 = tf.keras.applications.MobileNetV2(weights=None, classes=2)
