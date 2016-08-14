# Some common python image utilities

import numpy as np
from scipy.ndimage import filters
import collections

Gradients = collections.namedtuple('Gradients', ['imx', 'imy', 'mag', 'ori'])
# 
# Get image gradients, orientation, and magnitude of an image
# 
# @param image Image to get the gradients for
# @param sigma Sigma parameter for the gaussian filter
# @return Image gradient in the x direction
# @return Image gradient in the y direction
# @return mag Magnitude of the gradients
# @return ori Orientation of the gradients
# 
def get_derivative_orientation_and_mag(image, sigma=1):
	imx = np.zeros(image.shape)
	filters.gaussian_filter(image, (sigma, sigma), (0,1), imx)

	imy = np.zeros(image.shape)
	filters.gaussian_filter(image, (sigma, sigma), (1,0), imy)

	mag = np.sqrt(np.power(imx, 2) + np.power(imy, 2)) # Element-wise

	ori = np.arctan2(imy, imx) # Element-wise

	return Gradients(imx=imx, imy=imy, mag=mag, ori=ori)