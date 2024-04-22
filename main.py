#!/usr/bin/python3
import os
from pathlib import Path
from finder import *
import numpy as np

base = Path('../../../../').resolve() # should be vscode/CXFEL
images_dir = Path(base, 'images').resolve() # should be vscode/CXFEL/images
overlay_dir = Path(images_dir, 'peaks_water_overlay', '01').resolve() # should be vscode/CXFEL/images/peaks_water_overlay/01

print(f"Base directory: {base}\nImages directory: {images_dir}\nOverlay directory: {overlay_dir}\n")

files = [str(f) for f in overlay_dir.glob('*.h5') if f.name.startswith('overlay')]
# print(files)

rand = np.random.randint(0, len(files))
test_image, image_path = functions.load_h5(files[rand])

# ImageProcessor
ip = imageprocessor.ImageProcessor(test_image, 100)

# # using skimage: peak_local_max
coordinates = ip.find_peaks()
print(coordinates)

## visualize
ip.visualize_peaks()

# default background subtraction demo
b = background.BackgroundSubtraction(threshold=100)
dataframe = b.main(inputs=files) # input either tensor(s) or list of image paths
print(dataframe)

# ip = imageprocessor.ImageProcessor(test_image, 100)

# print(ip.coordinates)

# # 
# dataframe = b.main(files)
# print(dataframe)

# view waterbackground 
# functions.display_peaks_3d(b.loaded_image, b.coordinates, b.p.threshold_value)
# functions.display_peaks_3d_beamstop(b.loaded_image, b.p.threshold_value)

