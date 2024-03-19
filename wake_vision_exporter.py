# Use this script to export wake vision annotations to a csv file

import tensorflow_datasets as tfds
import csv
from tqdm import tqdm
import wake_vision_loader
import data_filters

from experiment_config import get_cfg

# First load Open Images v7
cfg = get_cfg()

print("Loading Open Images v7...")
open_images_v7 = tfds.load(
    "partial_open_images_v7",
    data_dir=cfg.WV_DIR,
    shuffle_files=False,
)


def export_dataset(dataset_name, eval, cfg):
    # Convert Open Images to Wake Vision
    print(f"Converting Open Images {dataset_name} to Wake Vision...")
    wake_vision = wake_vision_loader.open_images_to_wv(
        open_images_v7[dataset_name], dataset_name, cfg
    )

    csv_columns = ["filename", "person"]

    image_dictionary = {}

    if cfg.LABEL_TYPE == "bbox":
        additional_csv_columns = ["depiction", "body_part"]
        csv_columns.extend(additional_csv_columns)

        # Get a dataset with information about depictions
        print("Setting up depiction filters...")
        assert cfg.EXCLUDE_DEPICTION_SKULL_FLAG == False
        wake_vision_non_person = wake_vision.filter(data_filters.non_person_filter)

        wake_vision_depictions_persons = wake_vision_non_person.filter(
            lambda ds_entry: data_filters.depiction_eval_filter(
                ds_entry, return_person_depictions=True
            )
        )

        wake_vision_depictions_non_persons = wake_vision_non_person.filter(
            lambda ds_entry: data_filters.depiction_eval_filter(
                ds_entry, return_person_depictions=False
            )
        )

        # Get a dataset consisting only of person body parts
        print("Setting up body part filters...")
        body_parts_cfg = get_cfg("body_parts")
        body_parts_cfg.BBOX_PERSON_DIRECTORY = {}
        body_parts_cfg.BODY_PARTS_FLAG = True

        wake_vision_body_parts = wake_vision_loader.open_images_to_wv(
            open_images_v7[dataset_name], dataset_name, body_parts_cfg
        )

        if eval:
            additional_csv_columns = [
                "predominantly_female",
                "predominantly_male",
                "gender_unknown",
                "young",
                "middle_age",
                "older",
                "near",
                "medium_distance",
                "far",
                "dark",
                "normal_lighting",
                "bright",
                "person_depiction",
                "non-person_depiction",
                "non-person_non-depiction",
            ]
            csv_columns.extend(additional_csv_columns)

            print("Setting up gender MIAP filters...")
            wake_vision_female_person = data_filters.get_predominantly_female_set(
                wake_vision
            )

            wake_vision_male_person = data_filters.get_predominantly_male_set(
                wake_vision
            )
            wake_vision_gender_unknown = data_filters.get_unknown_gender_set(
                wake_vision
            )

            print("Setting up age MIAP filters...")
            wake_vision_young_person = data_filters.get_young_set(wake_vision)
            wake_vision_middle_person = data_filters.get_middle_set(wake_vision)
            wake_vision_old_person = data_filters.get_older_set(wake_vision)
            wake_vision_age_unknown = data_filters.get_unknown_age_set(wake_vision)

            print("Setting up distance filters...")
            distance_cfg = get_cfg("distance")
            distance_cfg.MIN_BBOX_SIZE = 0.001

            distance_wake_vision = wake_vision_loader.open_images_to_wv(
                open_images_v7[dataset_name], dataset_name, distance_cfg
            )

            wake_vision_far_persons = distance_wake_vision.filter(
                lambda ds_entry: data_filters.filter_bb_area(ds_entry, 0.001, 0.1)
            )
            wake_vision_mid_persons = distance_wake_vision.filter(
                lambda ds_entry: data_filters.filter_bb_area(ds_entry, 0.1, 0.6)
            )
            wake_vision_near_persons = distance_wake_vision.filter(
                lambda ds_entry: data_filters.filter_bb_area(ds_entry, 0.6, 100.0)
            )

            print("Setting up lighting filters...")
            wake_vision_dark = data_filters.get_low_lighting(wake_vision)
            wake_vision_normal_light = data_filters.get_medium_lighting(wake_vision)
            wake_vision_bright = data_filters.get_high_lighting(wake_vision)

            print("Setting up fine-grained depiction filters...")
            wake_vision_non_person_non_depiction = wake_vision_non_person.filter(
                lambda ds_entry: not data_filters.depiction_eval_filter(
                    ds_entry, return_all_depictions=True
                )
            )

    print("Writing person information...")
    for image in tqdm(wake_vision):
        filename = str(image["image/filename"].numpy())[1:].replace("'", "")
        image_dictionary[filename] = {}
        # Add whether a the image contains a person
        image_dictionary[filename]["person"] = image["person"].numpy()

    if cfg.LABEL_TYPE == "bbox":
        print("Writing person depiction information...")
        for image in tqdm(wake_vision_depictions_persons):
            filename = str(image["image/filename"].numpy())[1:].replace("'", "")
            image_dictionary[filename]["depiction"] = 1

        print("Writing non-person depiction information...")
        for image in tqdm(wake_vision_depictions_non_persons):
            filename = str(image["image/filename"].numpy())[1:].replace("'", "")
            image_dictionary[filename]["depiction"] = 1

        print("Writing body part information...")
        for image in tqdm(wake_vision_body_parts):
            filename = str(image["image/filename"].numpy())[1:].replace("'", "")
            image_dictionary[filename]["body_part"] = 1

        if eval:
            print("Writing gender MIAPs...")
            for image in tqdm(wake_vision_female_person):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["predominantly_female"] = 1

            for image in tqdm(wake_vision_male_person):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["predominantly_male"] = 1

            for image in tqdm(wake_vision_gender_unknown):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["gender_unknown"] = 1

            print("Writing age MIAPs...")
            for image in tqdm(wake_vision_young_person):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["young"] = 1

            for image in tqdm(wake_vision_middle_person):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["middle_age"] = 1

            for image in tqdm(wake_vision_old_person):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["older"] = 1

            for image in tqdm(wake_vision_age_unknown):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["age_unknown"] = 1

            print("Writing distance...")
            for image in tqdm(wake_vision_far_persons):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                # The far dataset may contain previously excluded images
                if filename in image_dictionary:
                    image_dictionary[filename]["far"] = 1
                else:
                    image_dictionary[filename] = {
                        "person": -1,
                        "depiction": -1,
                        "body_part": -1,
                        "predominantly_female": -1,
                        "predominantly_male": -1,
                        "gender_unknown": -1,
                        "young": -1,
                        "middle_age": -1,
                        "older": -1,
                        "near": -1,
                        "medium_distance": -1,
                        "far": 1,
                        "dark": -1,
                        "normal_lighting": -1,
                        "bright": -1,
                        "person_depiction": -1,
                        "non-person_depiction": -1,
                        "non-person_non-depiction": -1,
                    }

            for image in tqdm(wake_vision_mid_persons):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["medium_distance"] = 1

            for image in tqdm(wake_vision_near_persons):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["near"] = 1

            print("Writing lighting...")
            for image in tqdm(wake_vision_dark):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["dark"] = 1

            for image in tqdm(wake_vision_normal_light):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["normal_lighting"] = 1

            for image in tqdm(wake_vision_bright):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["bright"] = 1

            print("Writing fine-grained depiction...")

            for image in tqdm(wake_vision_depictions_persons):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["person_depiction"] = 1

            for image in tqdm(wake_vision_depictions_non_persons):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["non-person_depiction"] = 1

            for image in tqdm(wake_vision_non_person_non_depiction):
                filename = str(image["image/filename"].numpy())[1:].replace("'", "")
                image_dictionary[filename]["non-person_non-depiction"] = 1

    if cfg.LABEL_TYPE == "bbox" and eval:
        print(f"Exporting bbox and benchmark information to csv...")
        with open(f"tmp/wake_vision_{dataset_name}.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for filename in image_dictionary:
                writer.writerow(
                    {
                        "filename": filename,
                        "person": image_dictionary[filename]["person"],
                        "depiction": image_dictionary[filename].get("depiction", 0),
                        "body_part": image_dictionary[filename].get("body_part", 0),
                        "predominantly_female": image_dictionary[filename].get(
                            "predominantly_female", 0
                        ),
                        "predominantly_male": image_dictionary[filename].get(
                            "predominantly_male", 0
                        ),
                        "gender_unknown": image_dictionary[filename].get(
                            "gender_unknown", 0
                        ),
                        "young": image_dictionary[filename].get("young", 0),
                        "middle_age": image_dictionary[filename].get("middle_age", 0),
                        "older": image_dictionary[filename].get("older", 0),
                        "near": image_dictionary[filename].get("near", 0),
                        "medium_distance": image_dictionary[filename].get(
                            "medium_distance", 0
                        ),
                        "far": image_dictionary[filename].get("far", 0),
                        "dark": image_dictionary[filename].get("dark", 0),
                        "normal_lighting": image_dictionary[filename].get(
                            "normal_lighting", 0
                        ),
                        "bright": image_dictionary[filename].get("bright", 0),
                    }
                )

    elif cfg.LABEL_TYPE == "bbox":
        print(f"Exporting bbox information to csv...")
        with open(f"tmp/wake_vision_{dataset_name}.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for filename in image_dictionary:
                writer.writerow(
                    {
                        "filename": filename,
                        "person": image_dictionary[filename]["person"],
                        "depiction": image_dictionary[filename].get("depiction", 0),
                        "body_part": image_dictionary[filename].get("body_part", 0),
                    }
                )
    else:
        print(f"Exporting image information to csv...")
        with open(f"tmp/wake_vision_{dataset_name}.csv", "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for filename in image_dictionary:
                writer.writerow(
                    {
                        "filename": filename,
                        "person": image_dictionary[filename]["person"],
                    }
                )


# Export the bbox training set
export_dataset("train", eval=False, cfg=cfg)

# Export the image training set
image_cfg = get_cfg("image")
image_cfg.LABEL_TYPE = "image"
export_dataset("train", eval=False, cfg=image_cfg)


# Export the validation set
export_dataset("validation", eval=True, cfg=cfg)

# Export the test set
export_dataset("test", eval=True, cfg=cfg)
