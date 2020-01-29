""" Taken from mgharbi's work on demosaic net"""

"""Utilities to make a mosaic mask and apply it to an image."""
import numpy as np


__all__ = ["bayer", "xtrans"]


"""Bayer mosaic.

The patterned assumed is::

  GR R
  b GB

Args:
  im (np.array): image to mosaic. Dimensions are [c, h, w]

Returns:
  np.array: mosaicked image 
"""
def bayer(im, flat=False, return_masks=False):
    red = 0
    green = 1
    blue = 2
    bayer_mosaic = np.zeros((1, im.shape[1], im.shape[2]), dtype=np.int16)
    bayer_mosaic[0,0::2,0::2] = im[green,0::2,0::2]
    bayer_mosaic[0,1::2,1::2] = im[green,1::2,1::2]
    bayer_mosaic[0,0::2,1::2] = im[red,0::2,1::2]
    bayer_mosaic[0,1::2,0::2] = im[blue,1::2,0::2]

    return bayer_mosaic
