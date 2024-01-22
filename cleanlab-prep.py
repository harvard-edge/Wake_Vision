import tensorflow_datasets as tfds
import tensorflow as tf
import os
from tqdm.auto import tqdm
import io
import zipfile
from PIL import Image

import wake_vision_loader
from experiment_config import default_cfg as cfg

ds = tfds.load(
    "partial_open_images_v7",
    data_dir=cfg.WV_DIR,
    shuffle_files=False,
)

wv_ds = wake_vision_loader.open_images_to_wv(ds["test"], cfg.COUNT_PERSON_SAMPLES_TEST)

label_mapping = {0: "No Person", 1: "Person"}


def format_tensorflow_image_dataset(dataset, label_mapping, filename, save_dir):
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

            buf = io.BytesIO()
            image.save(buf, format="PNG")
            image_data = buf.getvalue()

            yield f"tf_dataset/{label}/image_{idx}.png", image_data

    zip_path = os.path.join(save_dir, f"{filename}.zip")

    with zipfile.ZipFile(zip_path, "w") as zf:
        for path, data in tqdm(image_data_generator()):
            zf.writestr(path, data)

    print(f"Saved zip file to: {zip_path}")


format_tensorflow_image_dataset(
    dataset=wv_ds,
    label_mapping=label_mapping,
    filename="wv_bbox_test",
    save_dir="./tmp",
)
