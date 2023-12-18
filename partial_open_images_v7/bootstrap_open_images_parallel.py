# A parallel version of bootstrap_open_images.py. Mainly useful for processing the very large training set. Beware that this script may use a large amount of disk space as the tar archives seem to be created locally before being uploaded to the cloud.

from google.api_core.exceptions import ServiceUnavailable
import os
import multiprocessing
import time
import tarfile
import io
import tensorflow_datasets as tfds
import numpy as np
from etils import epath

SPLIT = "train"
TARGET_PIXELS = 200000
JPEG_QUALITY = 72

# This can be reduced in case not enough memory or storage is available on the device
FROM_PREFIX = 10
TO_PREFIX = 99  # Inclusive

prefixes = [f"{x}" for x in range(FROM_PREFIX, TO_PREFIX + 1)]


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
    _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return io.BytesIO(buff.tobytes())


def tar_blobs(prefix):
    # Create tar file
    with epath.Path(
        f"gs://wake-vision-storage/tensorflow_datasets/downloads/manual/wake-vision-{SPLIT}-{prefix}.tar"
    ).open("w") as fobj:
        tar_file = tarfile.TarFile(fileobj=fobj, mode="w")
        blob_path = epath.Path(
            f"gs://wake-vision-storage/tensorflow_datasets/downloads/manual/{SPLIT}"
        )
        for blob in blob_path.glob(f"{prefix}*.jpg"):
            while True:
                try:
                    blob_path = f"/tmp/{os.path.basename(blob.name)}"

                    reduced_image = _resize_image_if_necessary(blob, TARGET_PIXELS)

                    with open(blob_path, "wb") as file_object:
                        file_object.write(reduced_image.getbuffer())

                    tar_file.add(blob_path, arcname=os.path.basename(blob.name))
                    os.remove(blob_path)
                    break
                except ServiceUnavailable:
                    time.sleep(1)
        tar_file.close()


processor_pool = multiprocessing.Pool()
processor_pool.map(tar_blobs, prefixes)
