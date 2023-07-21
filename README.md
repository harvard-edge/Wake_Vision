# VWWv2
A repository to work on a second edition of the Visual Wake Words Dataset

Currently, the repository contains a script to download the Open Images V4 Dataset and set up a basic vww-2 dataset using TensorFlow Datasets. 
In addition, the script also trains a basic mobilenet version1 model on this dataset, achieving around 90% accuracy on our runs. 
The requirements.txt file contains the Python packages required to run the script.

The script is resource intensive and will therefore have trouble running in anything but a cloud or high-performance computing machine. 
When running the script, we reached a maximum RAM consumption of 538 GB, and the disk usage of this repository after running the script was 1.4 TB.

We experienced several “Connection Reset by Peer” exceptions that caused the script to crash during the Open Images V4 Dataset download. 
We have yet to find the reason for this, but simply restarting the script when it crashes due to this exception will continue the download.
