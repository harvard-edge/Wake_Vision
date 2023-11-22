# Use this file to collapse the folder hierarchy inside the splits of the Open Images dataset.

from google.cloud import storage
import os

SPLIT = "validation"


for blob in storage.Client().list_blobs(
    "wake-vision-storage",
    prefix=f"tensorflow_datasets/downloads/manual/{SPLIT}/",
    match_glob="**.jpg",
):
    # Download the image.
    image_fobj = open("temp.jpg", "wb")
    blob.download_to_file(image_fobj)
    image_fobj = open("temp.jpg", "rb")
    # Upload the image.
    new_blob = storage.Blob(
        f"tensorflow_datasets/downloads/manual/{SPLIT}/{os.path.basename(blob.name)}",
        blob.bucket,
    )
    new_blob.upload_from_file(image_fobj)
    blob.delete()
