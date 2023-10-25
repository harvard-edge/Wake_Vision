# This file contains a script to train a mobilenetv1 model on the wake vision dataset, report the test accuracy and save the model.

import tensorflow as tf

import wake_vision_creation
import mobilenet_v1_dataset_preprocessing as preprocessor
import mobilenet_v1_silican_labs

EPOCHS = 15  # Same as used to train the vww model (ML commons tiny)
BEST_WEIGHTS_PATH = "checkpoints/best_weights.ckpt"

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

mobilenetv1_wake_vision = mobilenet_v1_silican_labs.mobilenet_v1()

mobilenetv1_wake_vision.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=BEST_WEIGHTS_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    save_weights_only=True,
    mode="max",
)

mobilenetv1_wake_vision.fit(
    mobilenetv1_train,
    epochs=EPOCHS,
    verbose=2,
    validation_data=mobilenetv1_val,
    callbacks=[model_checkpoint_callback],
)

mobilenetv1_wake_vision.load_weights(BEST_WEIGHTS_PATH)

mobilenetv1_wake_vision.evaluate(mobilenetv1_test, verbose=2)

print(mobilenetv1_wake_vision.get_metrics_result())

# Save the model
mobilenetv1_wake_vision.save("models/mobilenet_v1_wake_vision.keras")
