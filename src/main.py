#!/usr/bin/python3
import os
import classes as c 
import background as bg
import overwrite as s

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# instance
dh = c.DataHandler(base)

# images
high = dh._load_h5_image('9_18_23_high_intensity_3e8keV-2.h5')
low = dh._load_h5_image('9_18_23_low_intensity_3e7keV-2.h5')

# ImageProcessor
ip_high = c.ImageProcessor(high, 100)
ip_low = c.ImageProcessor(low, 100)

high_stream, high_intensity, _ = dh._load_stream_data('test_high.stream')
low_stream, low_intensity, _  = dh._load_stream_data('test_low.stream')

# using skimage: peak_local_max
high_coordinates = ip_high.find_peaks()
low_coordinates = ip_low.find_peaks()

# visualize
ip_high.visualize_peaks()
ip_low.visualize_peaks()

ip_high.visualize_image_3d(high_coordinates, ip_high.threshold)
ip_low.visualize_image_3d(low_coordinates, ip_low.threshold)

# default background subtraction demo
b = bg.BackgroundSubtraction()
for r in b.radii:
    b._coordinate_menu(r)

# view waterbackground 
bg.display_peaks_3d(b.loaded_image, b.coordinates, b.p.threshold_value)
bg.display_peaks_3d_beamstop(b.loaded_image, b.p.threshold_value)

# stream/overwrite adaptation demo
# USE DH

dh._overwrite(high_stream)


