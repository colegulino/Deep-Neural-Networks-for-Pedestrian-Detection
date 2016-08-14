# Library for generating and comparing histograms

import image_utils
import image_print_utils

import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import filters

# Here is a dictionary that specifies different openCV distance metrics that you can use
# for comparing histograms
comparison_methods = \
{
	'correlation' : cv2.cv.CV_COMP_CORREL,
	'chi-squared' : cv2.cv.CV_COMP_CHISQR,
	'intersection' : cv2.cv.CV_COMP_INTERSECT,
	'bhattacharyya' : cv2.cv.CV_COMP_BHATTACHARYYA	
} 

# 
# Function to ensure that color histogram parameters make sense
# 
# @param image Image to get the color histogram parameters for
# @param channels Color channels to get the histograms for
# @param bins Bins in each color channel (should be same length as channels)
# @param ranges Ranges for the values that are counted in the histogram (two for each channel)
# @param mask Mask on the image
# @exception Throws a ValueError if the shape of the parameters are invalid
# 
def check_color_hist_params(image, channels, bins, ranges, mask):
	if max(channels) > image.shape[0] - 1:
		raise ValueError("Max channel requested ({}) is larger than channels in image: {}".format(max(channels), image.shape[0]))
	if len(channels) != len(bins):
		raise ValueError("Length of channels ({}) does not equal length of bins ({}):".format(len(channels), len(bins)))
	if (2*len(channels)) != len(ranges):
		raise ValueError("Size of range ({}) is not appropriate size ({}).".format(len(ranges), 2*len(channels)))
	if mask.shape[:1] != image.shape[:1] and mask != None:
		raise ValueError("Shape of mask ({}) not same shape as image ({})".format(mask.shape[:1], image.shape[:1]))

# 
# Function that compares two histograms
# 
# @param hist_a One histogram to compare 
# @param hist_b One histogram to compare
# @param method_name String to determine what comparison method to use defined abvoe in comparison_methods
# @return A real value to describe the simularity between two histograms
# 
def compare_histograms(hist_a, hist_b, method_name='intersection'):
	return cv2.compareHist(hist_a, hist_b, comparison_methods[method_name])

# 
# Function to get an L1 normalized color histogram
# 
# @param image Image to get the color histogram parameters for
# @param channels Color channels to get the histograms for
# @param bins Bins in each color channel (should be same length as channels)
# @param ranges Ranges for the values that are counted in the histogram (two for each channel)
# @param mask Mask on the image
# @return A color histogram that is L1 normalized
# 
def get_normalized_histogram(image, channels=[0,1,2], bins=[25,25,25], ranges=[0,256,0,256,0,256], mask=None):
	check_color_hist_params(image, channels, bins, ranges, mask)
	hist = cv2.calcHist([image], channels, mask, bins, ranges)
	hist = cv2.normalize(hist).flatten()
	return hist

# 
# Function to get a non-normalized color histogram
# 
# @param image Image to get the color histogram parameters for
# @param channels Color channels to get the histograms for
# @param bins Bins in each color channel (should be same length as channels)
# @param ranges Ranges for the values that are counted in the histogram (two for each channel)
# @param mask Mask on the image
# @return A color histogram
# 
def get_histogram(image, channels=[0,1,2], bins=[25,25,25], ranges=[0,256,0,256,0,256], mask=None):
	check_color_hist_params(image, channels, bins, ranges, mask)
	return cv2.calcHist([image], channels, mask, bins, ranges)

# 
# Function to get separate RGB histograms for a color image
# 
# @param image Image to get the color histogram parameters for
# @param bins Bins in each color channel (should be same length as channels)
# @param ranges Ranges for the values that are counted in the histogram (two for each channel)
# @param mask Mask on the image
# @return Three histograms that represent RGB histograms respectively
# 
def get_rgb_histograms(image, bins=[256], ranges=[0,256], mask=None):
	# Example from: http://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html#gsc.tab=0
	colors = ('b', 'g', 'r')
	histograms = {}

	for i, color in enumerate(colors):
		histograms[color] = get_normalized_histogram(image, [i], bins, ranges, mask)

	b_hist = histograms['b']
	g_hist = histograms['g']
	r_hist = histograms['r']

	return r_hist, g_hist, b_hist

# 
# Function to get the HoG features of an image patch
# Examples: http://stackoverflow.com/questions/28390614/opencv-hogdescripter-python
# 
# @param image Image patch to get HoG descriptors for | Shape: (w,h,3)
# @param sigma Sigma parameter for the gaussian filter gradient
# @param no_bins Number of orientation bins
# @param th Threshold parameter for magnitude threshold between (0,1)
# @param mask Binary mask for the image
# @return Histogram of gradient orientations of the image
# 
def get_sift_features(image, sigma=1, no_bins=10, th=0.1, mask=None):
	# Apply mask if there is one
	im = np.copy(image)
	# if mask != None:
	if True:
		for i in range(3):
			im[:,:,i] = mask * image[:,:,i]

	shape = im.shape
	channels = shape[2]

	# Get gradients, orientations and magnitudes of gradients
	gradients = {}
	thresholds = {}
	for channel in range(channels):
		gradients[channel] = image_utils.get_derivative_orientation_and_mag(im[:,:,channel], sigma)
		thresholds[channel] = th * np.max(gradients[channel].mag)

	# get orientation ranges for each bin
	orientations = []
	step = (np.pi - (-np.pi)) / no_bins
	for ori_bin in np.arange(-np.pi, np.pi, step):
		orientations.append(ori_bin)
	orientations.append(np.pi)

	# A histogram for each color channel
	histograms = {}
	for channel in range(channels):
		histograms[channel] = np.zeros((shape[0], shape[1], no_bins))

	# Get the histograms for each pixel location
	for ori_bin in range(len(orientations) - 1):
		curr_range = (orientations[ori_bin], orientations[ori_bin + 1])
		for channel in range(channels):
			hist = histograms[channel]
			hist[:,:,ori_bin] = (gradients[channel].ori > curr_range[0]).astype(int) & \
								(gradients[channel].ori < curr_range[1]).astype(int) & \
								(gradients[channel].mag > thresholds[channel]).astype(int)

	# Sum up the histograms at each pixel and concatenate the histograms from each channel
	hist = np.array([]).astype(np.float32)
	for channel in range(channels):		
		hist = np.append(hist, np.sum(histograms[channel], axis=(0,1)))

	if(np.linalg.norm(hist) == 0):
		print("Norm of the histogram = 0. Histogram is invalid.")

	hist = hist / np.linalg.norm(hist)

	return np.array(hist).astype(np.float32)

if __name__ == "__main__":
	print("Testing histogram utilities.")

	# image_name = "large_pedestrian.jpg"
	image_name = "treeAndHorse.jpg"
	im = Image.open(image_name)
	im = im.convert('RGB')
	im = np.array(im)

	hist = get_sift_features(im)

	print("L1 Norm of Histogram: {}".format(np.linalg.norm(hist)))

	image_print_utils.print_histogram(hist)