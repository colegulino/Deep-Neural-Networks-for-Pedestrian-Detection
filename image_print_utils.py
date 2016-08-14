import image_segmentation
import simularity_set
import histogram_utils

from PIL import Image
from pylab import *
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt 

import numpy as np
import cv2

# 
# Function that prints a segmented image in one frame, and a bounding box around a region.
# The image is masked out so that only the region is shown and the bounding box
# 
# @param region Region you want to draw the bounding box on
# @param region_image Image that contains the classes at each index
# @param sim_set Simularity set that contains the bounding boxes
# @param seg_image Image that has colors for each region
# 
def print_bounding_box_region_and_seg_image(region, region_image, sim_set, seg_image):
	bbox = sim_set.create_bounding_box(region)
	class_mask = (region_image == region).astype(int)

	fig = figure()
	seg_image_full = fig.add_subplot(2, 1, 1)
	seg_image_show = seg_image_full.imshow(seg_image)
	seg_im = seg_image.convert('RGB')
	seg_im = np.array(seg_im)
	for i in range(3):
		seg_im[:,:,i] = class_mask * seg_im[:,:,i]

	seg_image_mask = fig.add_subplot(2, 1, 2)

	seg_im = Image.fromarray(seg_im)
	seg_image_mask.imshow(seg_im)
	rect = Rectangle((bbox.x0, bbox.y0), bbox.width, bbox.height, ec='red', fill=False)
	seg_image_mask.add_patch(rect)
	show()

# 
# Print the color histogram of a region in an image
# 
# @param region Region of an image to calculate the histogram for
# @param image Image to calculate histogram for
# @param sim_set Simularity set that contains the region information
# 
def print_region_histogram(region, image, sim_set):
	# Example from: http://docs.opencv.org/3.1.0/d1/db7/tutorial_py_histogram_begins.html#gsc.tab=0
	colors = ('b', 'g', 'r')

	mask = sim_set.get_region_mask(region)

	r_hist, g_hist, b_hist = histogram_utils.get_rgb_histograms(image, [256], [0, 256], mask)

	plt.plot(r_hist, 'r')
	plt.plot(g_hist, 'g')
	plt.plot(b_hist, 'b')
	plt.show()

# 
# Plot a histogram 
# 
# @param hist Histogram to plot
# @param color Color of the plot
# @param title Title of the plot
# @param xlabel Independent axis label
# @param ylabel Dependent axis label
# 
def print_histogram(hist, color='r', title='Histogram', xlabel='Bin', ylabel='Number'):
	plt.plot(hist, color)
	plt.xlim([0, len(hist)])
	plt.ylim([0, np.max(hist)])
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()
