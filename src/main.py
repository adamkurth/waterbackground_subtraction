#!/usr/bin/python3
import os
from pathlib import Path
from finder import *

base = Path('../').resolve() # cxfel
images_dir = Path('../images').resolve() # cxfel/images
print(f"Base directory: {base}")
print(f'Images directory: {images_dir} \n')

# instance
dh = datahandler.DataHandler(base)

# images)
high_image = dh.load_h5_image(image_name='9_18_23_high_intensity_3e8keV-2.h5')
low_image = dh.load_h5_image(image_name='9_18_23_low_intensity_3e7keV-1.h5')

# ImageProcessor
ip_high = imageprocessor.ImageProcessor(high_image, 100)
ip_low = imageprocessor.ImageProcessor(low_image, 100)

high_stream, high_intensity, _ = dh.load_stream_data('test_high.stream')
low_stream, low_intensity, _  = dh.load_stream_data('test_low.stream')

# using skimage: peak_local_max
high_coordinates = ip_high.find_peaks()
low_coordinates = ip_low.find_peaks()

# # visualize
# ip_high.visualize_peaks()
# ip_low.visualize_peaks()
# ip_high.visualize_image_3d(high_coordinates, ip_high.threshold)
# ip_low.visualize_image_3d(low_coordinates, ip_low.threshold)

# default background subtraction demo
b = background.BackgroundSubtraction()
for r in b.radii:
    b.coordinate_menu(r)

data = b.coordinate_menu_streamlined()
print(data)

# view waterbackground 
# functions.display_peaks_3d(b.loaded_image, b.coordinates, b.p.threshold_value)
# functions.display_peaks_3d_beamstop(b.loaded_image, b.p.threshold_value)

# only uncomment if want to overwrite stream
# dh.overwrite(high_stream)


