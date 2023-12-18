This is a dataset builder for the 7th version of the Open Images Dataset. At this point the builder only supports a part of the dataset, including image-level labels, bounding box labels and MAIP labels.

The builder currently relies on all images being downloaded into the tensorflow datasets manual_dir according to the instructions found here: https://github.com/cvdfoundation/open-images-dataset#download-full-dataset-with-google-storage-transfer.

Afterwards run the collapse_open_images_heirarchy script for each split (configured in the script), followed by the bootstrap_open_images.py script for each split (configured in the script). For the train split it may be beneficial to run the bootstrapping in parallel using either the bootstrap_open_images_parallel.py script or the bootstrap_open_images_parallel.sh script.

After running those script the dataset can be built as any other tensorflow dataset.