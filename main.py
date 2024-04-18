#!/usr/bin/python3
import os
from pathlib import Path
from finder import *
import numpy as np

base = Path('../../../../../').resolve() # should be vscode/CXFEL
images_dir = Path(base, 'images').resolve() # should be vscode/CXFEL/images
print(f"Base directory: {base}")
print(f'Images directory: {images_dir} \n') 

files = [str(f) for f in images_dir.glob('*.h5') if f.name.startswith('img')]

# tesing peaks in cxls_hitfinder
peaks_dir = Path(base, 'cxls_hitfinder', 'images', 'peaks', '01').resolve()
print(f'Peaks directory: {peaks_dir} \n')

peaks_image = [f for f in peaks_dir.glob('*.h5') if f.name.startswith('img')][0] 
peaks_image, _ = functions.load_h5(peaks_image)
print(f'Test image shape: {peaks_image.shape}')
np.save('/Users/adamkurth/Documents/vscode/CXFEL/cxls_hitfinder/cnn/src/pkg/waterbackground_subtraction/peaks.npy', peaks_image)


# test_image, image_path = functions.load_h5(images_dir) #random 

# ImageProcessor
# ip = imageprocessor.ImageProcessor(test_image, 100)

# # using skimage: peak_local_max
# coordinates = ip.find_peaks()

## visualize
# ip.visualize_peaks()

# default background subtraction demo
b = background.BackgroundSubtraction()
# original method
# for r in b.radii:
#     b.coordinate_menu(r)

# streamlined method
# dataframe = b.coordinate_menu_streamlined()


# test_image, image_path = functions.load_h5(files[0])
# np.save('/Users/adamkurth/Documents/vscode/CXFEL/cxls_hitfinder/cnn/src/pkg/waterbackground_subtraction/test_image.npy', test_image)

# ip = imageprocessor.ImageProcessor(test_image, 100)

# print(ip.coordinates)

# # 
# dataframe = b.main(files)
# print(dataframe)

# view waterbackground 
# functions.display_peaks_3d(b.loaded_image, b.coordinates, b.p.threshold_value)
# functions.display_peaks_3d_beamstop(b.loaded_image, b.p.threshold_value)

