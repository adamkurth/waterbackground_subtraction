#!/usr/bin/python3
import os
from pathlib import Path
from finder import *

base = Path('../').resolve() # cxfel
images_dir = Path('../images').resolve() # cxfel/images
print(f"Base directory: {base}")
print(f'Images directory: {images_dir} \n')

test_image, image_path = functions.load_h5(images_dir) #random 

# ImageProcessor
ip = imageprocessor.ImageProcessor(test_image, 100)

# using skimage: peak_local_max
coordinates = ip.find_peaks()

## visualize
# ip.visualize_peaks()

# default background subtraction demo
b = background.BackgroundSubtraction()
# original method
# for r in b.radii:
#     b.coordinate_menu(r)
# streamlined method
dataframe = b.coordinate_menu_streamlined()
print(dataframe)

# view waterbackground 
# functions.display_peaks_3d(b.loaded_image, b.coordinates, b.p.threshold_value)
# functions.display_peaks_3d_beamstop(b.loaded_image, b.p.threshold_value)
