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

# Read image
img = cv2.imread(r'000807.jpg')
# Make grey image
img = findpeaks.stats.togray(img)
# Scale between [0-255]
# img = findpeaks.stats.scale(img)

image_frost = findpeaks.frost_filter(img, damping_factor=2.0, win_size=7)
print("------")
img1 = Image.fromarray(image_frost, 'L')
img1.save(r"Frost_denoising.jpg")
img1.show()