import tensorflow as tf
import tensorflow_datasets as tfds

# Config
DATA_DIR = "./dataset/"
COUNT_PERSON_SAMPLES_TRAIN = 844965  # Number of person samples in the train sdataset. The number of non-person samples are 898077. We will use this number to balance the dataset.
COUNT_PERSON_SAMPLES_VAL = 9973  # There are 31647 non-person samples.
COUNT_PERSON_SAMPLES_TEST = 30226  # There are 95210 non-person samples. The distribution of persons in both the Val and Test set is close to 24% (Val:23.96) (Test:24.09) so we may not need to reduce the size of these.

# Start of function definitions


# A function to convert the "Train", "Validation" and "Test" parts of open images to their respective vww2 variants.
def open_images_to_vww2(ds_split, count_person_samples):
    # Use the image level classes already in the open images dataset to label images as containing a person or no person
    ds_split = ds_split.map(label_person, num_parallel_calls=tf.data.AUTOTUNE)

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


def mobilenetv1_preprocessing(ds_split):
    # Crop images to the resolution expected by mobilenet
    ds_split = ds_split.map(crop_images, num_parallel_calls=tf.data.AUTOTUNE)

    # Convert values from int8 to float32
    ds_split = ds_split.map(cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE)

    # Use the official mobilenet preprocessing to normalize images
    ds_split = ds_split.map(
        mobilenet_preprocessing_wrapper, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert each dataset entry from a dictionary to a tuple of (img, label) to be used by the keras API.
    ds_split = ds_split.map(prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE)

    # Batch and prefetch the dataset for improved performance
    return ds_split.batch(96).prefetch(tf.data.AUTOTUNE)


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


# End of function definitions

ds = tfds.load(
    "open_images_v4/200k",
    data_dir=DATA_DIR,
    shuffle_files=True,
)

ds["train"] = open_images_to_vww2(ds["train"], COUNT_PERSON_SAMPLES_TRAIN)
ds["validation"] = open_images_to_vww2(ds["validation"], COUNT_PERSON_SAMPLES_VAL)
ds["test"] = open_images_to_vww2(ds["test"], COUNT_PERSON_SAMPLES_TEST)

mobilenetv1_train = mobilenetv1_preprocessing(ds["train"])
mobilenetv1_val = mobilenetv1_preprocessing(ds["validation"])
mobilenetv1_test = mobilenetv1_preprocessing(ds["test"])


## Start of inference testing part
# Try out training and inference with mobilenet v1
# Parameters given to models are the same as for the models used in the visual wake words paper

# Optimizer
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.045, decay_steps=10000, decay_rate=0.98
)
optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, momentum=0.9)

# Loss
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# The visual wake words paper mention that the depth multiplier of their mobilenet model is 0.25.
# This is however not a possible value for the depth multiplier parameter of this api. There may be some termonology problems here where what the paper calls depth multiplier is the alpha parameter of the api.
mobilenetv1 = tf.keras.applications.MobileNet(alpha=0.25, weights=None, classes=2)

mobilenetv1.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

mobilenetv1.fit(
    mobilenetv1_train, epochs=10, verbose=2, validation_data=mobilenetv1_val
)

mobilenetv1.evaluate(mobilenetv1_test, verbose=2)

print(mobilenetv1.get_metrics_result())
