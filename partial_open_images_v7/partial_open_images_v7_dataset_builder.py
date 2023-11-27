"""Partial Open Images v7 dataset."""
from __future__ import annotations


import tensorflow_datasets as tfds
import os
import io
import csv
import collections
import numpy as np
from etils import epath
import functools
from typing import List

MAX_PIXELS = 200000
JPEG_QUALITY = 72


# Image Ids
TRAIN_IMAGE_IDS = "https://storage.googleapis.com/openimages/v6/oidv6-train-images-with-labels-with-rotation.csv"
VAL_IMAGE_IDS = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
TEST_IMAGE_IDS = "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv"


# Image level label source
TRAIN_HUMAN_LABELS = "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv"
TRAIN_MACHINE_LABELS = "https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-machine-imagelabels.csv"
VAL_HUMAN_LABELS = "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv"
VAL_MACHINE_LABELS = "https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-machine-imagelabels.csv"
TEST_HUMAN_LABELS = "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv"
TEST_MACHINE_LABELS = "https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-machine-imagelabels.csv"


# Bounding Box Label Source
TRAIN_BBOX_LABELS = (
    "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"
)
VAL_BBOX_LABELS = (
    "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
)
TEST_BBOX_LABELS = (
    "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"
)

IMAGE_LEVEL_SOURCES = [
    "verification",
    "crowdsource-verification",  # human labels
    "machine",
]

BBOX_SOURCES = [
    "freeform",
    "xclick",  # Manually drawn boxes.
    "activemil",  # Machine generated, human controlled.
]

_Object = collections.namedtuple("Object", ["label", "confidence", "source"])
_Bbox = collections.namedtuple(
    "Bbox",
    [
        "label",
        "source",
        "bbox",
        "is_occluded",
        "is_truncated",
        "is_group_of",
        "is_depiction",
        "is_inside",
    ],
)

_URLS = {
    "train_image_ids": TRAIN_IMAGE_IDS,
    "test_image_ids": TEST_IMAGE_IDS,
    "validation_image_ids": VAL_IMAGE_IDS,
    "train_human_labels": TRAIN_HUMAN_LABELS,
    "train_machine_labels": TRAIN_MACHINE_LABELS,
    "test_human_labels": TEST_HUMAN_LABELS,
    "test_machine_labels": TEST_MACHINE_LABELS,
    "validation_human_labels": VAL_HUMAN_LABELS,
    "validation_machine_labels": VAL_MACHINE_LABELS,
    "train-annotations-bbox": TRAIN_BBOX_LABELS,
    "test-annotations-bbox": TEST_BBOX_LABELS,
    "validation-annotations-bbox": VAL_BBOX_LABELS,
}


class Builder(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for partial_open_images_v7 dataset."""

    MANUAL_DOWNLOAD_INSTRUCTIONS = """
        Please use the instructions provided at: https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer to download the dataset images into tensorflow_datasets/downloads/manual/
        """

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        source_class_label = tfds.features.ClassLabel(
            names=IMAGE_LEVEL_SOURCES + BBOX_SOURCES
        )
        all_class_label = tfds.features.ClassLabel(
            names_file="/home/emjn/vww-2/partial_open_images_v7/all-classes.txt"
        )
        boxable_class_label = tfds.features.ClassLabel(
            names_file=tfds.core.tfds_path(
                os.path.join("object_detection", "open_images_classes_boxable.txt")
            )
        )
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(
                        shape=(None, None, 3),
                        doc="Image from the Open Images v7 dataset",
                    ),
                    "image/filename": tfds.features.Text(doc="Image filename"),
                    "objects": tfds.features.Sequence(
                        {
                            "label": all_class_label,
                            # Original data is 0, .1, ..., 1. We use 0, 1, 2, ..., 10.
                            "confidence": np.int32,
                            "source": source_class_label,
                        }
                    ),
                    "bobjects": tfds.features.Sequence(
                        {
                            "label": boxable_class_label,
                            "source": source_class_label,
                            "bbox": tfds.features.BBoxFeature(),
                            # Following values can be:
                            # 1 (true), 0 (false) and -1 (unknown).
                            "is_occluded": np.int8,
                            "is_truncated": np.int8,
                            "is_group_of": np.int8,
                            "is_depiction": np.int8,
                            "is_inside": np.int8,
                        }
                    ),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=None,  # Set to `None` to disable
            homepage="https://storage.googleapis.com/openimages/web/index.html",
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""

        paths = dl_manager.download_and_extract(_URLS)

        # TODO(partial_open_images_v7): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(
                dl_manager.manual_dir / "train", "train", paths
            ),
            "validation": self._generate_examples(
                dl_manager.manual_dir / "validation", "validation", paths
            ),
            "test": self._generate_examples(
                dl_manager.manual_dir / "test", "test", paths
            ),
        }

    def _generate_examples(self, path, split, paths):
        """Yields examples."""

        def load(names):
            csv_positions = [0] * len(names)
            return functools.partial(
                _load_objects, [paths[name] for name in names], csv_positions
            )

        def load_boxes(name):
            csv_positions = [0]
            return functools.partial(_load_bboxes, paths[name], csv_positions)

        objects = load([f"{split}_human_labels", f"{split}_machine_labels"])

        bboxes = load_boxes(f"{split}-annotations-bbox")

        objects = objects(None)
        bboxes = bboxes(None)

        url_to_image_id = _load_image_ids(paths[f"{split}_image_ids"])

        for f in path.glob("*.jpg"):
            file_name = os.path.basename(f)
            image_id = int(os.path.splitext(url_to_image_id[file_name])[0], 16)
            image_objects = [obj._asdict() for obj in objects.get(image_id, [])]
            image_objects = [
                image_object
                for image_object in image_objects
                if image_object["confidence"] >= 5
            ]
            image_bboxes = [bbox._asdict() for bbox in bboxes.get(image_id, [])]
            yield str(image_id), {
                "image": _resize_image_if_necessary(f, target_pixels=MAX_PIXELS),
                "image/filename": file_name,
                "objects": image_objects,
                "bobjects": image_bboxes,
            }


def _resize_image_if_necessary(image_fobj, target_pixels=None):
    if target_pixels is None:
        return image_fobj

    cv2 = tfds.core.lazy_imports.cv2
    # Decode image using OpenCV2.
    image = cv2.imdecode(
        np.frombuffer(image_fobj.read_bytes(), dtype=np.uint8), flags=3
    )
    # Get image height and width.
    height, width, _ = image.shape
    actual_pixels = height * width
    if actual_pixels > target_pixels:
        factor = np.sqrt(target_pixels / actual_pixels)
        image = cv2.resize(image, dsize=None, fx=factor, fy=factor)
    # Encode the image with quality=72 and store it in a BytesIO object.
    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
    return io.BytesIO(buff.tobytes())


def _read_csv_line(line: bytes) -> List[str]:
    # Using `f.tell()` and csv.reader causes: `OsError: telling
    # position disabled by next() call`. So we read every line separately.
    csv_line = csv.reader([line.decode()])
    return next(csv_line)


def _load_image_ids(csv_path):
    images = {}
    with epath.Path(csv_path).open(mode="rb") as csv_f:
        csv_f.readline()  # Drop headers
        for line in csv_f:
            image_id, _, original_url, _, _, _, _, _, _, _, _, _ = _read_csv_line(line)
            original_url = os.path.basename(original_url)
            images[original_url] = image_id
    return images


def _load_objects(csv_paths, csv_positions, prefix):
    """Returns objects listed within given CSV files."""
    objects = collections.defaultdict(list)
    for i, labels_path in enumerate(csv_paths):
        with epath.Path(labels_path).open(mode="rb") as csv_f:
            if csv_positions[i] > 0:
                csv_f.seek(csv_positions[i])
            else:
                csv_f.readline()  # Drop headers
            for line in csv_f:
                (image_id, source, label, confidence) = _read_csv_line(line)
                if prefix and image_id[0] != prefix:
                    break
                csv_positions[i] = csv_f.tell()
                image_id = int(image_id, 16)
                current_obj = _Object(label, int(float(confidence) * 10), source)
                objects[image_id].append(current_obj)
    return dict(objects)


def _load_bboxes(csv_path, csv_positions, prefix):
    """Returns bounded boxes listed within given CSV file."""
    boxes = collections.defaultdict(list)
    with epath.Path(csv_path).open(mode="rb") as csv_f:
        if csv_positions[0] > 0:
            csv_f.seek(csv_positions[0])
        else:
            csv_f.readline()  # Drop headers
        if (
            str(os.path.basename(csv_path))
            == "openimages_v6_oidv6-train-annotations-bboxo4dXxAZfPjuOYCY-C3GFHEiwJr2M5GKQxCQk8Gj3-cY.csv"
        ):
            for line in csv_f:
                (
                    image_id,
                    source,
                    label,
                    confidence,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    is_occluded,
                    is_truncated,
                    is_group_of,
                    is_depiction,
                    is_inside,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = _read_csv_line(line)
                if prefix and image_id[0] != prefix:
                    break
                csv_positions[0] = csv_f.tell()
                image_id = int(image_id, 16)
                del confidence  # always 1 in bounding boxes.
                current_row = _Bbox(
                    label,
                    source,
                    tfds.features.BBox(
                        float(ymin), float(xmin), float(ymax), float(xmax)
                    ),
                    int(is_occluded),
                    int(is_truncated),
                    int(is_group_of),
                    int(is_depiction),
                    int(is_inside),
                )
                boxes[image_id].append(current_row)
        else:
            for line in csv_f:
                (
                    image_id,
                    source,
                    label,
                    confidence,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    is_occluded,
                    is_truncated,
                    is_group_of,
                    is_depiction,
                    is_inside,
                ) = _read_csv_line(line)
                if prefix and image_id[0] != prefix:
                    break
                csv_positions[0] = csv_f.tell()
                image_id = int(image_id, 16)
                del confidence  # always 1 in bounding boxes.
                current_row = _Bbox(
                    label,
                    source,
                    tfds.features.BBox(
                        float(ymin), float(xmin), float(ymax), float(xmax)
                    ),
                    int(is_occluded),
                    int(is_truncated),
                    int(is_group_of),
                    int(is_depiction),
                    int(is_inside),
                )
                boxes[image_id].append(current_row)
    return dict(boxes)
