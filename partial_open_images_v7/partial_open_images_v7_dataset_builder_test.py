import partial_open_images_v7_dataset_builder
import tensorflow_datasets as tfds


class PartialOpenImagesV7(tfds.testing.DatasetBuilderTestCase):
    DATASET_CLASS = partial_open_images_v7_dataset_builder.Builder
    SPLITS = {
        "train": 8,  # Number of fake train examples. One file does not result in a sample.
        "validation": 5,  # Number of fake validation examples
        "test": 5,  # Number of fake test examples
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}

    DL_EXTRACT_RESULT = {
        "train_image_ids": "train_images.csv",
        "test_image_ids": "test-images-with-rotation.csv",
        "validation_image_ids": "validation-images-with-rotation.csv",
        "train_human_labels": "train_human_labels.csv",
        "train_machine_labels": "train_machine_labels.csv",
        "test_human_labels": "oidv7-test-annotations-human-imagelabels.csv",
        "test_machine_labels": "oidv7-test-annotations-machine-imagelabels.csv",
        "validation_human_labels": "oidv7-val-annotations-human-imagelabels.csv",
        "validation_machine_labels": "oidv7-val-annotations-machine-imagelabels.csv",
        "train-annotations-bbox": "openimages_v6_oidv6-train-annotations-bbox.csv",
        "test-annotations-bbox": "test-annotations-bbox.csv",
        "validation-annotations-bbox": "validation-annotations-bbox.csv",
        "validation-annotations-miap": "open_images_extended_miap_boxes_val.csv",
        "test-annotations-miap": "open_images_extended_miap_boxes_test.csv",
    }


if __name__ == "__main__":
    tfds.testing.test_main()
