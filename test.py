#!/usr/bin/python3
import os
from pathlib import Path
from finder import *
import numpy as np

b = background.BackgroundSubtraction(threshold=100)
b.test_demo()