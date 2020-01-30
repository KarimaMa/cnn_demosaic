import numpy as np
from PIL import Image
from bayer import bayer
import argparse
import os
import random

def dump(x):
    if len(x.shape) == 3:
        print(x[:,0:5,0:5])
    else:
        print(x[0:5,0:5])


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
            im = np.array(pic, dtype=np.uint8)
            im = np.transpose(im, (2, 0, 1))

            file_prefix = filename.split(".")[0] 
            out_basedir = os.path.join(args.output_dir, file_prefix)
            if not os.path.exists(out_basedir):
                os.mkdir(out_basedir)

            bayer_mosaic = np.zeros((4, img_w//2, img_w//2), dtype=np.uint8)
            bayer_mosaic[0,:,:] = im[0,0::2,1::2]
            bayer_mosaic[1,:,:] = im[1,0::2,0::2]
            bayer_mosaic[2,:,:] = im[1,1::2,1::2]
            bayer_mosaic[3,:,:] = im[2,1::2,0::2]

            missing_bayer = np.zeros((8, img_w//2, img_w//2), dtype=np.uint8)
            missing_bayer[0,:,:] = im[1,0::2,1::2] # g at r
            missing_bayer[1,:,:] = im[2,0::2,1::2] # b at r
            missing_bayer[2,:,:] = im[0,0::2,0::2] # r at gr
            missing_bayer[3,:,:] = im[2,0::2,0::2] # b at gr
            missing_bayer[4,:,:] = im[0,1::2,1::2] # r at gb
            missing_bayer[5,:,:] = im[2,1::2,1::2] # b at gb
            missing_bayer[6,:,:] = im[0,1::2,0::2] # r at b
            missing_bayer[7,:,:] = im[1,1::2,0::2] # g at b

            bayer_mosaic.tofile(os.path.join(out_basedir, "dense_bayer.data"))
            im.tofile(os.path.join(out_basedir, "image.data"))
            missing_bayer.tofile(os.path.join(out_basedir, "missing_bayer.data"))

            if (args.debug):
                print("bayer mosaic")
                dump(bayer_mosaic)
                print("image")
                dump(im)
                print("missing")
                dump(missing_bayer)
                exit()
