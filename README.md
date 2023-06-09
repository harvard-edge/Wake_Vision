# VWWv2
A repository to work on a second edition of the Visual Wake Words Dataset

Currently, the repository contains a script to download the Open Images Dataset V4 using TensorFlow Datasets and a pip requirements file listing the python packages needed to run the script.

The script is resource intensive and will therefore have trouble running in anything but a cloud or high performance computing machine.
When running the script I reached a maximum RAM consumption of 538 GB, and the disk usage of this repository after running the script is 1.4 TB.

I experienced several “Connection Reset by Peer” exceptions while running the script crashes when running the script. 
I have yet to find the reason for this, but simply restarting the script when it crashes due to this exception will continue the download.