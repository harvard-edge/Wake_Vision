import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import wake_vision_loader
from experiment_config import default_cfg as cfg
import finer_grained_evaluation_filters as fgef
from wake_vision_loader import (
    person_filter,
    non_person_filter,
    label_person_image_labels,
    label_person_bbox_labels,
)

# Get the dataset we want
orig_ds = tfds.load("partial_open_images_v7", data_dir=cfg.WV_DIR, shuffle_files=True)

ds = orig_ds["train"]

# Label Persons

# full_length = 0
# for _ in ds:
#    full_length += 1
# print(f"Full Length: {full_length}")

# Count DS samples
# person_length = 0
# for _ in person_ds:
#    person_length += 1
# print(f"Person DS Length: {person_length}")
# non_person_length = 0
# for _ in non_person_ds:
#    non_person_length += 1
# print(f"Non-Person DS Length: {non_person_length}")

# print(f"Excluded: {full_length - person_length - non_person_length}")


# Take 5 examples from the dataset
gen_items = [
    {
        "image": cv2.cvtColor(
            tf.image.convert_image_dtype(sample["image"], tf.uint8).numpy(),
            cv2.COLOR_RGB2BGR,
        ),
        "person": label_person_image_labels(sample)["person"],
    }
    for i, sample in enumerate(ds.take(20))
]

# Show the images
for i, item in enumerate(gen_items):
    print("Image", i)
    print("Person:", item["person"])
    print("------------------")
    cv2.imwrite(f"tmp/{i}.png", item["image"])
