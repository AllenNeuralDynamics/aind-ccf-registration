"""
Converts a file - test file
"""

import numpy as np
import tifffile
from skimage import io

inputfile = "output_newsubject.tiff"
outputfile = "output_new_subject16.tiff"

im = io.imread(inputfile)
print(np.max(im), np.min(im), np.mean(im), np.std(im))
im = im.astype(np.uint16)
print(np.max(im), np.min(im), np.mean(im), np.std(im))
tifffile.imwrite(outputfile, im)

# flake8: noqa: E501
ref = tifffile.imread(
    "/Users/sharmishtaas/Desktop/expts/brainreg_jack/templates/allen_mouse_25um_v1.2/reference.tiff"
)
print(type(ref[0][0][0]))
