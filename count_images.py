# A script to count the amount of samples (images) in the different Wake Vision splits.
import wake_vision_loader
import tensorflow_datasets as tfds
import experiment_config
import data_filters


def count_reduce(old_state, input_element):
    return old_state + 1


def count_miaps(cfg, split):
    miaps = wake_vision_loader.get_miaps(cfg, batch_size=1, split=split)

    print(f"MIAPS {split} split")
    print(f"Female: {miaps['female'].reduce(0, count_reduce).numpy()}")
    print(f"Male: {miaps['male'].reduce(0, count_reduce).numpy()}")
    print(f"Gender Unknown: {miaps['gender_unknown'].reduce(0, count_reduce).numpy()}")
    print(f"Young: {miaps['young'].reduce(0, count_reduce).numpy()}")
    print(f"Middle: {miaps['middle'].reduce(0, count_reduce).numpy()}")
    print(f"Older: {miaps['older'].reduce(0, count_reduce).numpy()}")
    print(f"Age Unknown: {miaps['age_unknown'].reduce(0, count_reduce).numpy()}")
    print("\n\n")


def count_lightning(cfg, split):
    lighting = wake_vision_loader.get_lighting(cfg, batch_size=1, split=split)

    print(f"Lighting {split} split")
    print(f"Dark: {lighting['dark'].reduce(0, count_reduce).numpy()}")
    print(f"Normal Light: {lighting['normal_light'].reduce(0, count_reduce).numpy()}")
    print(f"Bright: {lighting['bright'].reduce(0, count_reduce).numpy()}")
    print("\n\n")


def count_distance(cfg, split):
    distance = wake_vision_loader.get_distance_eval(cfg, batch_size=1, split=split)

    print(f"Distance {split} split")
    print(f"Near: {distance['near'].reduce(0, count_reduce).numpy()}")
    print(f"Middle: {distance['mid'].reduce(0, count_reduce).numpy()}")
    print(f"Far: {distance['far'].reduce(0, count_reduce).numpy()}")
    print("\n\n")


def count_hand_foot(cfg, split):
    hand_feet = wake_vision_loader.get_hands_feet_eval(cfg, batch_size=1, split=split)

    print(f"Hand and Feet {split} split")
    print(f"Human Hand: {hand_feet['Human hand'].reduce(0, count_reduce).numpy()}")
    print(f"Human Foot: {hand_feet['Human foot'].reduce(0, count_reduce).numpy()}")
    print("\n\n")


def count_depiction(cfg, split):
    depiction = wake_vision_loader.get_depiction_eval(cfg, batch_size=1, split=split)

    print(f"Depiction {split} split")
    print(
        f"Person Depictions: {depiction['depictions_persons'].reduce(0, count_reduce).numpy()}"
    )
    print(
        f"Non-Person Depictions: {depiction['depictions_non_persons'].reduce(0, count_reduce).numpy()}"
    )
    print(
        f"Non-Person Non-Depictions: {depiction['non_person_no_depictions'].reduce(0, count_reduce).numpy()}"
    )
    print("\n\n")


def count_full_ds(cfg):
    ds = tfds.load(
        "partial_open_images_v7",
        data_dir=cfg.WV_DIR,
        shuffle_files=False,
    )

    train_split = ds["train"]
    validation_split = ds["validation"]
    test_split = ds["test"]

    full_train_len = len(train_split)
    full_validation_len = len(validation_split)
    full_test_len = len(test_split)

    train_split = wake_vision_loader.open_images_to_wv(train_split, "train", cfg)
    validation_split = wake_vision_loader.open_images_to_wv(
        validation_split, "validation", cfg
    )
    test_split = wake_vision_loader.open_images_to_wv(test_split, "test", cfg)

    train_person = train_split.filter(data_filters.person_filter)
    train_non_person = train_split.filter(data_filters.non_person_filter)

    validation_person = validation_split.filter(data_filters.person_filter)
    validation_non_person = validation_split.filter(data_filters.non_person_filter)

    test_person = test_split.filter(data_filters.person_filter)
    test_non_person = test_split.filter(data_filters.non_person_filter)

    print("Full dataset")

    train_person_count = train_person.reduce(0, count_reduce).numpy()
    train_non_person_count = train_non_person.reduce(0, count_reduce).numpy()
    train_exclude_count = full_train_len - train_person_count - train_non_person_count
    print(
        f"Train - Person: {train_person_count} Non-Person: {train_non_person_count} Excluded {train_exclude_count}"
    )

    val_person_count = validation_person.reduce(0, count_reduce).numpy()
    val_non_person_count = validation_non_person.reduce(0, count_reduce).numpy()
    val_exclude_count = full_validation_len - val_person_count - val_non_person_count
    print(
        f"Validation - Person: {val_person_count} Non-Person: {val_non_person_count} Excluded {val_exclude_count}"
    )

    test_person_count = test_person.reduce(0, count_reduce).numpy()
    test_non_person_count = test_non_person.reduce(0, count_reduce).numpy()
    test_exclude_count = full_test_len - test_person_count - test_non_person_count
    print(
        f"Test - Person: {test_person_count} Non-Person: {test_non_person_count} Excluded {test_exclude_count}"
    )
    print("\n\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--full_dataset", action="store_true")
    parser.add_argument("-m", "--miaps_dataset", action="store_true")
    parser.add_argument("-l", "--lighting_dataset", action="store_true")
    parser.add_argument("-d", "--distance_dataset", action="store_true")
    parser.add_argument("-hf", "--hand_feet_dataset", action="store_true")
    parser.add_argument("-p", "--depiction_dataset", action="store_true")
    parser.add_argument(
        "-la", "--label_type", type=str, default="bbox", choices=["bbox", "image"]
    )

    args = parser.parse_args()

    cfg = experiment_config.get_cfg("count_config")
    cfg.LABEL_TYPE = args.label_type

    if args.full_dataset:
        count_full_ds(cfg)

    if args.miaps_dataset:
        count_miaps(cfg, "validation")
        count_miaps(cfg, "test")

    if args.lighting_dataset:
        count_lightning(cfg, "validation")
        count_lightning(cfg, "test")

    if args.distance_dataset:
        count_distance(cfg, "validation")
        count_distance(cfg, "test")

    if args.hand_feet_dataset:
        count_hand_foot(cfg, "validation")
        count_hand_foot(cfg, "test")

    if args.depiction_dataset:
        count_depiction(cfg, "validation")
        count_depiction(cfg, "test")
