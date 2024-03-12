import csv
import tensorflow as tf
import tensorflow_datasets as tfds

from experiment_config import default_cfg
import pp_ops
import partial_open_images_v7.partial_open_images_v7_dataset_builder
import data_filters


# A function to convert the "Train", "Validation" and "Test" parts of open images to their respective wake vision variants.
def open_images_to_wv(
    ds_split,
    split_name,
    cfg=default_cfg,
):
    # First use the config flags to figure out what labels should be considered as person labels.
    if cfg.LABEL_TYPE == "image":
        image_level_person_label_list = list(cfg.IMAGE_LEVEL_PERSON_DICTIONARY.values())
        if cfg.BODY_PARTS_FLAG:
            image_level_person_label_list.extend(
                cfg.IMAGE_LEVEL_BODY_PART_DICTIONARY.values()
            )
    bbox_person_label_list = list(cfg.BBOX_PERSON_DICTIONARY.values())
    if cfg.BODY_PARTS_FLAG:
        bbox_person_label_list.extend(cfg.BBOX_BODY_PART_DICTIONARY.values())

    # Use either the image level labels or bounding box labels (according to configuration) already in the open images dataset to label images as containing a person or no person
    if split_name == "train":
        if cfg.LABEL_TYPE == "image":
            ds_split = ds_split.map(
                lambda ds_entry: label_person_image_labels(
                    ds_entry, image_level_person_label_list, cfg=cfg
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        elif cfg.LABEL_TYPE == "bbox":
            ds_split = ds_split.map(
                lambda ds_entry: label_person_bbox_labels(
                    ds_entry, bbox_person_label_list, cfg=cfg
                ),  # pass cfg to function
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            raise ValueError(
                'Configuration option "Label Type" must be "image" or "bbox" for the Wake Vision Dataset.'
            )
    elif split_name == "validation" or split_name == "test":
        if cfg.LABEL_TYPE == "image" or cfg.LABEL_TYPE == "bbox":
            ds_split = ds_split.map(
                lambda ds_entry: label_person_bbox_labels(
                    ds_entry, bbox_person_label_list, cfg=cfg
                ),  # pass cfg to function
                num_parallel_calls=tf.data.AUTOTUNE,
            )
        else:
            raise ValueError(
                'Configuration option "Label Type" must be "image" or "bbox" for the Wake Vision Dataset.'
            )
    else:
        raise ValueError(
            'Encountered a split that was neither "train", "validation" or "test"'
        )

    # Correct labels found to be wrong using cleanlab. Currently we only have verified labels for the "validation" and "test" split.
    if split_name != "train":
        try:
            (
                verified_person_list,
                verified_non_person_list,
                verified_exclude_list,
                verified_depiction_list,
            ) = read_cleanlab_csv(f"cleaned_csvs/wv_{split_name}_cleaned.csv")
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find the file wv_{split_name}_cleaned.csv in the cleaned_csvs directory. Please download this file from the github repository, or generate it yourself using the scripts in the cleanlab_cleaning directory"
            )
        ds_split = ds_split.map(
            lambda ds_entry: correct_label_issues(
                ds_entry,
                verified_person_list,
                verified_non_person_list,
                verified_exclude_list,
                verified_depiction_list,
                cfg,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    # Filter the dataset into a part with persons and a part with no persons
    person_ds = ds_split.filter(data_filters.person_filter)
    non_person_ds = ds_split.filter(data_filters.non_person_filter)

    # We now interleave these two datasets to create a dataset that contains both examples of persons and no persons.
    # Choice dataset is a dataset that determines which dataset to sample from at each iteration. In our case, we want to sample from the person_ds and non_person_ds alternating each iteration.
    choice_ds = tf.data.Dataset.range(2).repeat()
    ds_split = tf.data.Dataset.choose_from_datasets(
        [person_ds, non_person_ds], choice_ds, stop_on_empty_dataset=True
    )

    return ds_split


def label_person_image_labels(ds_entry, person_label_list, cfg=default_cfg):
    if tf.reduce_any(
        list(
            data_filters.check_image_level_label(ds_entry, person_label, cfg)
            for person_label in person_label_list
        )
    ):
        ds_entry["person"] = 1
    # If a person related label is present but no person related label has passed the confidence threshold requirement to be labelled a person, we exclude the image.
    elif tf.logical_or(
        tf.reduce_any(
            list(
                tf.equal(
                    tf.constant(person_label, tf.int64),
                    ds_entry["objects"]["label"],
                )
                for person_label in person_label_list
            )
        ),
        (
            tf.logical_and(
                cfg.EXCLUDE_DEPICTION_SKULL_FLAG,
                tf.reduce_any(
                    list(
                        tf.equal(
                            tf.constant(skull_label, tf.int64),
                            ds_entry["objects"]["label"],
                        )
                        for skull_label in cfg.IMAGE_LEVEL_SKULL_DICTIONARY.values()
                    )
                ),
            )
        ),
    ):
        ds_entry["person"] = -1
    else:
        ds_entry["person"] = 0
    return ds_entry


def label_person_bbox_labels(ds_entry, person_label_list, cfg=default_cfg):
    if tf.math.equal(tf.size(ds_entry["bobjects"]["label"]), 0):
        ds_entry["person"] = -1
    elif tf.reduce_any(
        list(
            data_filters.check_bbox_label(ds_entry, person_label, cfg=cfg)
            for person_label in person_label_list
        )  # Person label that is not a depiction inside crop
    ):
        ds_entry["person"] = 1
    elif tf.reduce_any(
        list(
            data_filters.check_bbox_label(
                ds_entry, person_label, cfg=cfg, exclude_outside_crop=False
            )
            for person_label in person_label_list
        )  # Person label that is not a depiction outside crop
    ):
        ds_entry["person"] = -1
    elif tf.reduce_any(
        list(
            data_filters.check_bbox_label(
                ds_entry,
                person_label,
                cfg=cfg,
                exclude_depiction=False,
                exclude_outside_crop=False,
            )
            for person_label in (
                person_label_list + list(cfg.BBOX_SKULL_DICTIONARY.values())
            )
        )  # Person label that is a depiction or a skull
    ):
        ds_entry["person"] = -1 if cfg.EXCLUDE_DEPICTION_SKULL_FLAG else 0
    else:
        ds_entry["person"] = 0
    return ds_entry


def read_cleanlab_csv(file_path):
    # Initialize lists for each category
    verified_person_list = []
    verified_non_person_list = []
    verified_exclude_list = []
    verified_depiction_list = []

    # Open and read the CSV file
    with open(file_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # Check the value of 'clean_label' and append the filename to the respective list
            if row["clean_label"] == "1":
                verified_person_list.append(row["filename"])
            elif row["clean_label"] == "0":
                verified_non_person_list.append(row["filename"])
            elif row["clean_label"] == "-1":
                verified_exclude_list.append(row["filename"])
            elif row["clean_label"] == "-2":
                verified_depiction_list.append(row["filename"])
            else:
                raise ValueError(
                    "Encountered a clean_label that was not 1, 0, -1 or -2 in the cleanlab csv file."
                )

    # Return the lists
    return (
        verified_person_list,
        verified_non_person_list,
        verified_exclude_list,
        verified_depiction_list,
    )


def correct_label_issues(
    ds_entry,
    verified_person_list,
    verified_non_person_list,
    verified_exclude_list,
    verified_depiction_list,
    cfg,
):
    if tf.reduce_any(
        tf.equal(
            ds_entry["image/filename"],
            verified_person_list,
        )
    ):
        ds_entry["person"] = 1
    elif tf.reduce_any(
        tf.equal(
            ds_entry["image/filename"],
            verified_non_person_list,
        )
    ):
        ds_entry["person"] = 0
    elif tf.reduce_any(
        tf.equal(
            ds_entry["image/filename"],
            verified_exclude_list,
        )
    ):
        ds_entry["person"] = -1
    elif tf.reduce_any(
        tf.equal(
            ds_entry["image/filename"],
            verified_depiction_list,
        )
    ):
        if cfg.EXCLUDE_DEPICTION_SKULL_FLAG:
            ds_entry["person"] = -1
        else:
            ds_entry["person"] = 0
    return ds_entry


def preprocessing(ds_split, batch_size, train=False, cfg=default_cfg):
    # Convert values from int8 to float32
    ds_split = ds_split.map(
        pp_ops.cast_images_to_float32, num_parallel_calls=tf.data.AUTOTUNE
    )

    if train:
        # Repeat indefinitely and shuffle the dataset
        ds_split = ds_split.repeat().shuffle(cfg.SHUFFLE_BUFFER_SIZE)
        # inception crop
        ds_split = ds_split.map(
            pp_ops.inception_crop, num_parallel_calls=tf.data.AUTOTUNE
        )
        # resize
        resize = lambda ds_entry: pp_ops.resize(ds_entry, cfg.INPUT_SHAPE)
        ds_split = ds_split.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
        # flip
        ds_split = ds_split.map(
            pp_ops.random_flip_lr, num_parallel_calls=tf.data.AUTOTUNE
        )
    else:
        # resize small
        resize_small = lambda ds_entry: pp_ops.resize_small(ds_entry, cfg.INPUT_SHAPE)
        ds_split = ds_split.map(resize_small, num_parallel_calls=tf.data.AUTOTUNE)
        # center crop
        center_crop = lambda ds_entry: pp_ops.center_crop(ds_entry, cfg.INPUT_SHAPE)
        ds_split = ds_split.map(center_crop, num_parallel_calls=tf.data.AUTOTUNE)
        
    if cfg.grayscale:
        ds_split = ds_split.map(
            pp_ops.grayscale, num_parallel_calls=tf.data.AUTOTUNE
        )

    # Use the official mobilenet preprocessing to normalize images
    ds_split = ds_split.map(
        pp_ops.mobilenet_preprocessing_wrapper, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Convert each dataset entry from a dictionary to a tuple of (img, label) to be used by the keras API.
    ds_split = ds_split.map(
        pp_ops.prepare_supervised, num_parallel_calls=tf.data.AUTOTUNE
    )

    # Batch and prefetch the dataset for improved performance
    return ds_split.batch(batch_size).prefetch(2)


def get_wake_vision(cfg=default_cfg, batch_size=None):
    batch_size = batch_size or cfg.BATCH_SIZE
    ds = tfds.load(
        "partial_open_images_v7",
        data_dir=cfg.WV_DIR,
        shuffle_files=False,
    )

    ds["train"] = open_images_to_wv(ds["train"], "train", cfg=cfg)
    ds["validation"] = open_images_to_wv(ds["validation"], "validation", cfg=cfg)
    ds["test"] = open_images_to_wv(ds["test"], "test", cfg=cfg)

    train = preprocessing(ds["train"], batch_size, train=True, cfg=cfg)
    val = preprocessing(ds["validation"], batch_size, cfg=cfg)
    test = preprocessing(ds["test"], batch_size, cfg=cfg)

    return train, val, test


def get_lighting(cfg=default_cfg, batch_size=None, split="test"):
    if split != "train" and split != "validation" and split != "test":
        raise ValueError("Split must be 'train', 'validation, or 'test'")

    batch_size = batch_size or cfg.BATCH_SIZE
    ds = tfds.load(
        "partial_open_images_v7",
        data_dir=cfg.WV_DIR,
        shuffle_files=False,
        split=split,
    )

    wv_ds = open_images_to_wv(ds, split, cfg=cfg)
    
    #first filter persons and non-persons
    person = wv_ds.filter(data_filters.person_filter)
    non_person = wv_ds.filter(data_filters.non_person_filter)

    lighting_ds = {
        "person_dim": data_filters.get_low_lighting(person),
        "person_normal_light": data_filters.get_medium_lighting(person),
        "person_bright": data_filters.get_high_lighting(person),
        "non_person_dim": data_filters.get_low_lighting(non_person),
        "non_person_normal_light": data_filters.get_medium_lighting(non_person),
        "non_person_bright": data_filters.get_high_lighting(non_person),
    }

    for key, value in lighting_ds.items():
        lighting_ds[key] = preprocessing(value, batch_size, cfg=cfg)

    return lighting_ds


def get_miaps(cfg=default_cfg, batch_size=None, split="test"):
    if split != "test" and split != "validation":
        raise ValueError("split must be 'test' or 'validation'")

    batch_size = batch_size or cfg.BATCH_SIZE
    ds = tfds.load(
        "partial_open_images_v7",
        data_dir=cfg.WV_DIR,
        shuffle_files=False,
        split=split,
    )

    wv_ds = open_images_to_wv(ds, split, cfg=cfg)

    # Create finer grained evaluation sets before preprocessing the dataset.
    miaps = {
        "female": data_filters.get_predominantly_female_set(wv_ds),
        "male": data_filters.get_predominantly_male_set(wv_ds),
        "gender_unknown": data_filters.get_unknown_gender_set(wv_ds),
        "young": data_filters.get_young_set(wv_ds),
        "middle": data_filters.get_middle_set(wv_ds),
        "older": data_filters.get_older_set(wv_ds),
        "age_unknown": data_filters.get_unknown_age_set(wv_ds),
        "no_person": wv_ds.filter(data_filters.non_person_filter),
    }

    for key, value in miaps.items():
        miaps[key] = preprocessing(value, batch_size, cfg=cfg)

    return miaps


# Distance Eval
def get_distance_eval(cfg=default_cfg, batch_size=None, split="test"):
    if split != "test" and split != "validation":
        raise ValueError("split must be 'test' or 'validation'")

    batch_size = batch_size or cfg.BATCH_SIZE
    ds = tfds.load(
        "partial_open_images_v7",
        data_dir=cfg.WV_DIR,
        shuffle_files=False,
        split=split,
    )

    ds = open_images_to_wv(ds, split, cfg=cfg)
    no_person = ds.filter(data_filters.non_person_filter)
    far = ds.filter(
        lambda ds_entry: data_filters.filter_bb_area(ds_entry, 0.001, 0.1)
    )  # cfg.NEAR_BB_AREA))
    mid = ds.filter(
        lambda ds_entry: data_filters.filter_bb_area(ds_entry, 0.1, 0.6)
    )  # cfg.MID_BB_AREA))
    near = ds.filter(
        lambda ds_entry: data_filters.filter_bb_area(ds_entry, 0.6, 100.0)
    )  # cfg.FAR_BB_AREA))

    no_person = preprocessing(no_person, batch_size, cfg=cfg)
    far = preprocessing(far, batch_size, cfg=cfg)
    mid = preprocessing(mid, batch_size, cfg=cfg)
    near = preprocessing(near, batch_size, cfg=cfg)

    return {"far": far, "mid": mid, "near": near, "no_person": no_person}


def get_depiction_eval(cfg=default_cfg, batch_size=None, split="test"):
    if split != "test" and split != "validation":
        raise ValueError("split must be 'test' or 'validation'")
    batch_size = batch_size or cfg.BATCH_SIZE
    
    depiction_cfg = cfg.copy_and_resolve_references()
    depiction_cfg.EXCLUDE_DEPICTION_SKULL_FLAG = False
    ds = tfds.load(
        "partial_open_images_v7",
        data_dir=cfg.WV_DIR,
        shuffle_files=False,
        split=split,
    )
    
    wv_ds = open_images_to_wv(ds, split, cfg=depiction_cfg)
    
    #first filter persons and non-persons
    person = wv_ds.filter(data_filters.person_filter)
    non_person = wv_ds.filter(data_filters.non_person_filter)
    
    #then filter out person and non-person depictions from the non_person set
    depictions_persons = non_person.filter(lambda ds_entry: 
                    data_filters.depiction_eval_filter(ds_entry, return_person_depictions=True))
    depictions_non_persons = non_person.filter(lambda ds_entry: 
                    data_filters.depiction_eval_filter(ds_entry, return_person_depictions=False))
    
    
    non_person_no_depictions = non_person.filter(lambda ds_entry: 
                    not data_filters.depiction_eval_filter(ds_entry, return_all_depictions=True))
    
    person = preprocessing(person, batch_size, cfg=cfg)
    depictions_persons = preprocessing(depictions_persons, batch_size, cfg=cfg)
    depictions_non_persons = preprocessing(depictions_non_persons, batch_size, cfg=cfg)
    non_person_no_depictions = preprocessing(non_person_no_depictions, batch_size, cfg=cfg)
    
    return({"person": person,
            'depictions_persons': depictions_persons,
            'depictions_non_persons': depictions_non_persons,
            'non_person_no_depictions': non_person_no_depictions})
    
    