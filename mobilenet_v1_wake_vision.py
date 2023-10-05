# This file contains a script to train a mobilenetv1 model on the wake vision dataset, report the test accuracy and save the model.

import tensorflow as tf

import wake_vision_creation
import mobilenet_v1_dataset_preprocessing as preprocessor
import mobilenet_v1_silican_labs


wake_vision = wake_vision_creation.create_wake_vision_ds()

mobilenetv1_train = preprocessor.mobilenetv1_preprocessing(wake_vision["train"])
mobilenetv1_val = preprocessor.mobilenetv1_preprocessing(wake_vision["validation"])
mobilenetv1_test = preprocessor.mobilenetv1_preprocessing(wake_vision["test"])


## Start of inference testing part
# Try out training and inference with mobilenet v1
# Parameters given to models are the same as for the models used in the visual wake words paper

# Optimizer
optimizer = tf.keras.optimizers.RMSprop()

# Loss
loss = tf.keras.losses.SparseCategoricalCrossentropy()

# The visual wake words paper mention that the depth multiplier of their mobilenet model is 0.25.
# This is however not a possible value for the depth multiplier parameter of this api. There may be some termonology problems here where what the paper calls depth multiplier is the alpha parameter of the api.
mobilenetv1_wake_vision = mobilenet_v1_silican_labs.mobilenet_v1()

mobilenetv1_wake_vision.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

mobilenetv1_wake_vision.fit(
    mobilenetv1_train, epochs=40, verbose=2, validation_data=mobilenetv1_val
)

mobilenetv1_wake_vision.evaluate(mobilenetv1_test, verbose=2)

print(mobilenetv1_wake_vision.get_metrics_result())

# Save the model
mobilenetv1_wake_vision.save("models/mobilenet_v1_wake_vision.keras")
