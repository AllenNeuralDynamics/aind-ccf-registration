"""
Registers only - test file
"""

import ants

# flake8: noqa: E501
referenceimage = "/Users/sharmishtaas/Desktop/expts/brainreg_jack/templates/allen_mouse_25um_v1.2/reference.tiff"
testimage = "output_new_subject16.tiff"

print("Reading reference image")
img1 = ants.image_read(referenceimage)
print("Reading test image")
img2 = ants.image_read(testimage)

reg12 = ants.registration(img1, img2, "Affine", reg_iterations=[100, 10, 0])
ants.image_write(reg12["warpedmovout"], "warpedimage.tiff")
