import numpy as np
from PIL import Image
from bayer import bayer
import argparse
import os
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Provide im directory')
    parser.add_argument('--image_list', type=str, help='list of ims to use')
    parser.add_argument('--input_dir', type=str, help='the path of the input im directory')
    parser.add_argument('--output_dir', type=str, help='the path of the output im directory')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    img_w = 128

    with open(args.image_list, "r") as f:
        for l in f:
            filename = l.strip()
            print(filename)
            pic = Image.open(os.path.join(args.input_dir, filename))
            im = np.array(pic)
            im = np.transpose(im, (2, 0, 1))

            file_prefix = filename.split(".")[0] 
            out_basedir = os.path.join(args.output_dir, file_prefix)
            if not os.path.exists(out_basedir):
                os.mkdir(out_basedir)

            bayer_mosaic = bayer(im)
            bayer_mosaic.tofile(os.path.join(out_basedir, "bayer.data"))
            im.tofile(os.path.join(out_basedir, "image.data"))

            if (args.debug):
                print(bayer_mosaic[0])
                print("R")
                print(im[0])
                print("G")
                print(im[1])
                print("B")
                print(im[2])

