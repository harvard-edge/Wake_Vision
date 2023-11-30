# Use this file to collapse the folder hierarchy inside the splits of the Open Images dataset.

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
