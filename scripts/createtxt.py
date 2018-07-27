#!/usr/bin/env python

# Assuming the images and the labels have the same name, this script creates data.txt for the dense data layer (input)

import os, sys, cv2
import numpy as np
from itertools import izip
from argparse import ArgumentParser
from collections import OrderedDict
from skimage.io import ImageCollection, imsave
from skimage.transform import resize

def make_parser():
    parser = ArgumentParser()
    parser.add_argument(
        'photo_dir',
        help="Directory containing all RGB camvid label images as PNGs"
    )
    parser.add_argument(
        'label_dir',
        help="""Directory to save grayscale label images.
        Output images have same basename as inputs so be careful not to
        overwrite original RGB labels""")
    return parser

if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    image_file = ImageCollection(os.path.join(args.photo_dir, "*"))
    #label_file = ImageCollection(os.path.join(args.label_dir, "*"))
    #labels = ["Me", "So", "Wo", "br"];
    for i, inpath in enumerate(image_file):
	with open("roughness_compute_bn.txt", "a") as text_file:
            text_file.write(image_file.files[i]+" ")
            #text_file.write(label_file.files[i]+" ")
            text_file.write(str(0)+" ")
