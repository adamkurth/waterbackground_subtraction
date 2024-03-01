import background as bg


# Demo of waterbackground subtraction and Bragg peak integration
b = bg.BackgroundSubtraction()
b.main() # runs _coordinate_menu() for radii list

print(f'Number of coordinates above threshold of {b.p.threshold_value}):\n {len(b.coordinates)}')

high_data, high_intensity, high_path = b._load_stream('test_high.stream')
low_data, low_intensity, low_path = b._load_stream('test_low.stream')


# print(f'List of found Bragg peaks:\n {b.peaks}') # doesn't work right 

