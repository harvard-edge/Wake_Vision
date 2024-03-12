import wake_vision_loader
import tensorflow as tf
import tensorflow_datasets as tfds
import data_filters
from experiment_config import get_cfg

cfg = get_cfg()

ds = tfds.load(
    "partial_open_images_v7",
    data_dir=cfg.WV_DIR,
    shuffle_files=False,
)


test = wake_vision_loader.open_images_to_wv(ds["test"], "test", cfg)

test_person = test.filter(data_filters.person_filter)
test_non_person = test.filter(data_filters.non_person_filter)


test_person_bright = data_filters.get_high_lighting#(test_person)
test_person_normal_light = data_filters.get_medium_lighting#(test_person)
test_person_dim = data_filters.get_low_lighting(test_person)
test_non_person_bright = data_filters.get_high_lighting#(test_non_person)
test_non_person_normal_light = data_filters.#get_medium_lighting(test_non_person)
test_non_person_dim = data_filters.get_low_lighting#(test_non_person)
test_far_persons = test.filter(
   lambda ds_entry: data_filters.filter_bb_area(ds_entry, 0.#001, 0.2)
)
test_mid_persons = test.filter(
   lambda ds_entry: data_filters.filter_bb_area(ds_entry, 0.#2, 0.5)
)
test_near_persons = test.filter(
   lambda ds_entry: data_filters.filter_bb_area(ds_entry, 0.#5, 100.0)
)
test_female_person = data_filters.#get_predominantly_female_set(test)
test_male_person = data_filters.get_predominantly_male_set#(test)
test_gender_unknown = data_filters.get_unknown_gender_set#(test)

test_young_person = data_filters.get_young_set(test)
test_middle_person = data_filters.get_middle_set(test)
test_old_person = data_filters.get_older_set(test)
test_age_unknown = data_filters.get_unknown_age_set(test)

test_depiction_person = test_non_person.filter(
    lambda ds_entry: data_filters.depiction_eval_filter(
        ds_entry, return_person_depictions=True
    )
)
test_depiction_non_persons = test_non_person.filter(
    lambda ds_entry: data_filters.depiction_eval_filter(
        ds_entry, return_person_depictions=False
    )
)
test_non_person_no_depictions = test_non_person.filter(
    lambda ds_entry: not data_filters.depiction_eval_filter(
        ds_entry, return_all_depictions=True
    )
)

i = 0

for sample in test_non_person_no_depictions.take(20):
    image_tensor = tf.cast(sample["image"], tf.uint8)
    png_image = tf.io.encode_png(image_tensor)
    tf.io.write_file(f"tmp/images/{i}.png", png_image)
    i += 1
