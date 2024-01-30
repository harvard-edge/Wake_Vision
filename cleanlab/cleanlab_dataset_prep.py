# This file converts our tensorflow dataset into a zip file that can be used by Cleanlab.


import tensorflow_datasets as tfds
import os
from tqdm.auto import tqdm
import io
from PIL import Image

import wake_vision_loader
from experiment_config import default_cfg as cfg

ds = tfds.load(
    "partial_open_images_v7",
    data_dir=cfg.WV_DIR,
    shuffle_files=False,
)

# Get the validation and test wake vision dataset to prepare for export.
wv_ds_validation = wake_vision_loader.open_images_to_wv(ds["validation"], cfg.COUNT_PERSON_SAMPLES_VAL)
wv_ds_test = wake_vision_loader.open_images_to_wv(ds["test"], cfg.COUNT_PERSON_SAMPLES_TEST)

label_mapping = {0: "No Person", 1: "Person"}


def format_tensorflow_image_dataset(dataset, save_dir, split_name):
    """Convert a Tensorflow dataset to Cleanlab Studio format.

    dataset: tf.data.Dataset
        Tensorflow dataset
    image_key: str
        column name for image in dataset
    label_key: str
        column name for label in dataset
    label_mapping: Dict[str, int]
        id to label str mapping
    filename: str
        filename for the zip file
    save_dir: str
        directory to save the zip file

    """

    def image_data_generator():
        """Generator to yield image data and its path in the zip file."""
        for idx, example in enumerate(dataset):
            image = Image.fromarray(example["image"].numpy())
            label = label_mapping[example["person"].numpy()]
            filename = example["image/filename"].numpy()

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_data = buf.getvalue()

            yield f"wv-image-folder/{split_name}/{label}/{filename}.png", image_data

    for path, data in tqdm(image_data_generator()):
        os.makedirs(
            os.path.join(save_dir, "wv-image-folder", split_name, label_mapping[0]), exist_ok=True
        )
        os.makedirs(
            os.path.join(save_dir, "wv-image-folder", split_name, label_mapping[1]), exist_ok=True
        )
        with open(os.path.join(save_dir, path), "wb") as f:
            f.write(data)

    print(f"Finished converting {split_name} TensorFlow Dataset to Image Folder Dataset")


# Prepare the validation dataset
format_tensorflow_image_dataset(
    dataset=wv_ds_validation,
    save_dir="./tmp",
    split_name="validation",
)

# Prepare the test dataset
format_tensorflow_image_dataset(
    dataset=wv_ds_test,
    save_dir="./tmp",
    split_name="test",
)
