#!/usr/bin/python3

import background as bg

# Demo of waterbackground subtraction and Bragg peak integration

high = bg.BackgroundSubtraction('9_18_23_high_intensity_3e8keV-2.h5', 'test_high.stream')
low = bg.BackgroundSubtraction('9_18_23_low_intensity_3e7keV-2.h5', 'test_low.stream')

high.main()
low.main()


print(f'Number of coordinates above threshold of {high.p.threshold_value}):\n {len(high.coordinates)}')
print(f'Number of coordinates above threshold of {low.p.threshold_value}):\n {len(low.coordinates)}')

# b._overwrite(b.high_data)    

# print(f'List of found Bragg peaks:\n {b.peaks}') # doesn't work right 

