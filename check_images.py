import tensorflow as tf
import tensorflow_datasets as tfds
import cv2
import wake_vision_loader
from experiment_config import default_cfg as cfg
import finer_grained_evaluation_filters as fgef

# Get the dataset we want
orig_ds = tfds.load(
    "partial_open_images_v7",
    data_dir=cfg.WV_DIR,
    shuffle_files=False,
)

ds = orig_ds["test"]
ds = wake_vision_loader.open_images_to_wv(ds, cfg.COUNT_PERSON_SAMPLES_TEST)

miaps_test = {
    "female": fgef.get_predominantly_female_set(ds),
    "male": fgef.get_predominantly_male_set(ds),
    "gender_unknown": fgef.get_unknown_gender_set(ds),
    "young": fgef.get_young_set(ds),
    "middle": fgef.get_middle_set(ds),
    "older": fgef.get_older_set(ds),
    "age_unknown": fgef.get_unknown_age_set(ds),
}

lighting_ds = {
    "dark": fgef.get_low_lighting(ds),
    "normal_light": fgef.get_medium_lighting(ds),
    "bright": fgef.get_high_lighting(ds),
}

ds = lighting_ds["dark"]

# Count DS samples
d = 0
for _ in ds:
    d += 1


# Take 5 examples from the dataset
gen_items = [
    {
        "image": cv2.cvtColor(
            tf.image.convert_image_dtype(sample["image"], tf.uint8).numpy(),
            cv2.COLOR_RGB2BGR,
        ),
        "original_link": sample["image/filename"],
        "miap_labels": sample["miaps"],
        "depiction": sample["bobjects"],
    }
    for i, sample in enumerate(ds.take(10))
]

# Show the images
for i, item in enumerate(gen_items):
    print("Image", i)
    print("MIAPS:", item["miap_labels"])
    print("Bobjects:", item["depiction"])
    print("------------------")
    cv2.imwrite(f"tmp/{i}.png", item["image"])


print(f"Length: {d}")
