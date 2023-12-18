# Use this file to collapse the folder hierarchy inside the splits of the Open Images dataset. This currently needs to run before the bootstrap_open_images* scripts as the etil library used in those files does not support the "**" glob pattern. This script may incur extra costs - and it may be possible to avoid them by using the google.cloud.storage library directly in the bootstrap_open_images* scripts.

from google.cloud import storage
import os

SPLIT = "validation"

source_bucket = storage.Client().bucket("wake-vision-storage")

for blob in source_bucket.list_blobs(
    prefix=f"tensorflow_datasets/downloads/manual/{SPLIT}/",
    match_glob="**.jpg",
):
    source_bucket.rename_blob(
        blob,
        f"tensorflow_datasets/downloads/manual/{SPLIT}/{os.path.basename(blob.name)}",
    )
