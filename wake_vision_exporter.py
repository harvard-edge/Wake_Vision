# Use this script to export wake vision annotations to a csv file

import tensorflow_datasets as tfds
import csv
from tqdm import tqdm
import wake_vision_loader

from experiment_config import default_cfg as cfg

# First load Open Images v7
print("Loading Open Images v7...")
open_images_v7 = tfds.load(
    "partial_open_images_v7",
    data_dir=cfg.WV_DIR,
    shuffle_files=False,
)

# Convert Open Images to Wake Vision
print("Converting Open Images to Wake Vision...")
wake_vision_train = wake_vision_loader.open_images_to_wv(
    open_images_v7["train"], "train", cfg
)
wake_vision_validation = wake_vision_loader.open_images_to_wv(
    open_images_v7["validation"], "validation", cfg
)
wake_vision_test = wake_vision_loader.open_images_to_wv(
    open_images_v7["test"], "test", cfg
)

# Export the training set
print("Exporting training set...")


def write_wake_vision_csv(split, split_name):
    with open(f"tmp/wake_vision_{split_name}.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "filename",
                "person",
            ]
        )
        for example in tqdm(split):
            writer.writerow(
                [
                    str(example["image/filename"].numpy())[1:].replace("'", ""),
                    example["person"].numpy(),
                ]
            )


write_wake_vision_csv(wake_vision_train, "train")

# Export the validation set
print("Exporting validation set...")
write_wake_vision_csv(wake_vision_validation, "validation")

# Export the test set
print("Exporting test set...")
write_wake_vision_csv(wake_vision_test, "test")
