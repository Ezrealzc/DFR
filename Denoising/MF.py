from pylab import *
import numpy as np
from PIL import ImageFilter
from PIL import Image
from PIL import ImageFilter
import cv2
from pylab import *
import scipy
from scipy.ndimage.filters import uniform_filter
from scipy.ndimage.measurements import variance
import findpeaks
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import bm3d

# Read image
img = cv2.imread(r'000807.jpg')
# Some pre-processing
# Make grey image
img = findpeaks.stats.togray(img)
# img = findpeaks.stats.scale(img)
# Scale between [0-255]
image_MF = findpeaks.median_filter(img, win_size=7)
img1 = Image.fromarray(image_MF, 'L')
img1.save(r"MF_denoising.jpg")
img1.show()






