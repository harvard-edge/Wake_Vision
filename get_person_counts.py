import tensorflow_datasets as tfds
import tensorflow as tf
from wake_vision_loader import label_person_bbox_labels, label_person_image_labels, person_filter
from experiment_config import get_cfg
from vww_loader import get_vww

# cfg = get_cfg("wv_small")
# cfg.LABEL_TYPE = "image"
# cfg.INPUT_SHAPE = (96, 96, 3)

# ds = tfds.load(
#         "partial_open_images_v7",
#         data_dir=cfg.WV_DIR,
#         shuffle_files=False,
#     )


# for split in ["train", "validation", "test"]:
#     ds_split = ds[split]

#     if cfg.LABEL_TYPE == "image":
#             ds_split = ds_split.map(
#                 label_person_image_labels, num_parallel_calls=tf.data.AUTOTUNE
#             )
#     elif cfg.LABEL_TYPE == "bbox":
        
#         ds_split = ds_split.map(
#             lambda ds_entry: label_person_bbox_labels(ds_entry, cfg=cfg), #pass cfg to function
#             num_parallel_calls=tf.data.AUTOTUNE
#         )
#     else:
#         raise ValueError(
#             'Configuration option "Label Type" must be "image" or "bbox" for the Wake Vision Dataset.'
#         )

#     # Filter the dataset into a part with persons and a part with no persons
#     person_ds = ds_split.filter(person_filter)
#     count_person_sample = 0
#     for entry in person_ds:
#         count_person_sample += 1
#     print(f"Count Person Samples {cfg.LABEL_TYPE} {split}: {count_person_sample}")


train, val, test = get_vww()
num_samples = 0
for entry in train:
    num_samples += 1
print("VWW Train Size: ", num_samples)

num_samples = 0
for entry in val:
    num_samples += 1
print("VWW Val Size: ", num_samples)

num_samples = 0
for entry in test:
    num_samples += 1
print("VWW Test Size: ", num_samples)