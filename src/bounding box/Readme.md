# Detection and Bounding Box

This code is run in google colab

***
### Command for running cli in google colab
> ### %%writefile parsing.py

### Import Useful libraries 
> ### import torch 
> ### import argparse
> ### import glob

####   This function is designed to do the detection process and draw 
####   a bounding box. Additionally, If the level of confidence for 
####   the detected image is equal to or higher than the passed confidence, 
####   the information about the box of the image will be displayed.



### CLI 
> !python3 parsing.py -c <level_of_confidence> --path <image_directory> -w <weight_of_trained_model>

