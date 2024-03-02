#!/usr/bin/python3
import os
import classes as c 
import background as bg
import overwrite as s

base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# instance
dh = c.DataHandler(base)


# images
high = dh.load_h5_image('9_18_23_high_intensity_3e8keV-2.h5')
low = dh.load_h5_image('9_18_23_low_intensity_3e7keV-2.h5')

# ImageProcessor
ip_high = c.ImageProcessor(high, 100)
ip_low = c.ImageProcessor(low, 100)

high_stream, high_intensity, _ = dh.load_stream_data('test_high.stream')
low_stream, low_intensity, _  = dh.load_stream_data('test_low.stream')

# using skimage: peak_local_max
high_coordinates = ip_high.find_peaks()
low_coordinates = ip_low.find_peaks()

# visualize
# ip_high.visualize_peaks()
# ip_low.visualize_peaks()

# ip_high.visualize_image_3d(high_coordinates, ip_high.threshold)
# ip_low.visualize_image_3d(low_coordinates, ip_low.threshold)

# default background subtraction demo
b = bg.BackgroundSubtraction()
# for r in b.radii:
#     b._coordinate_menu(r)

# view waterbackground 
# bg.display(b.loaded_image, b.coordinates, b.p.threshold_value)

bg.display_peaks_3d(b.loaded_image, b.coordinates, b.p.threshold_value)
bg.display_peaks_3d_beamstop(b.loaded_image, b.p.threshold_value)

# stream/overwrite adaptation demo
s = s.Stream()
high_data, high_intensity, high_path = s._load_stream('test_high.stream')
low_data, low_intensity, low_path = s._load_stream('test_low.stream')
# s._overwrite(s.high_data)    


# Demo of waterbackground subtraction and Bragg peak integration

# high = hl.BackgroundSubtraction('9_18_23_high_intensity_3e8keV-2.h5', 'test_high.stream')
# low = hl.BackgroundSubtraction('9_18_23_low_intensity_3e7keV-2.h5', 'test_low.stream')

# high.main()
# low.main()


# print(f'Number of coordinates above threshold of {high.p.threshold_value}):\n {len(high.coordinates)}')
# print(f'Number of coordinates above threshold of {low.p.threshold_value}):\n {len(low.coordinates)}')

# b._overwrite(b.high_data)    

# print(f'List of found Bragg peaks:\n {b.peaks}') # doesn't work right 

