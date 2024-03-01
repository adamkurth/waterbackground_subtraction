import background as bg


# demo of waterbackground subtraction and Bragg peak integration
b = bg.BackgroundSubtraction()
b.main() # runs _coordinate_menu() for radii list

print(f'Number of coordinates above threshold of {b.p.threshold_value}):\n {len(b.coordinates)}')

# print(f'List of found Bragg peaks:\n {b.peaks}') # doesn't work right 

