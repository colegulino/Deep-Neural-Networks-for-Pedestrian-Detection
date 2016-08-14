# Here is a general python script for running a prototype of the algorithm described in
# Effecient Graph-Based Image Segmentation by Felzenzwalb and Huttenlocher
# Link: https://cs.brown.edu/~pff/papers/seg-ijcv.pdf
# Source Code Link: http://cs.brown.edu/~pff/segment/ 

import image_segmentation
import simularity_set
import image_print_utils
import histogram_utils

from PIL import Image
from pylab import *
from matplotlib.patches import Rectangle

import numpy as np

if __name__ == "__main__":
	# Show the image before segmentation
	
	# image_name = "treeAndHorse.jpg"
	image_name = "large_pedestrian.jpg"
	im = Image.open(image_name)
	im = im.convert('RGB')
	im = np.array(im)

	# Parameters
	sigma = 0.5
	min_size = 100
	th = 3000 # Large th results in larget components

	seg_image, pixel_class, disjoint_set = image_segmentation.segment_image(im, sigma, th, min_size)

	print("Get simularity set")
	sim_set = simularity_set.simularity_set(region_image=pixel_class, image=im, disjoint_set=disjoint_set)

	count = 0
	reg_a = None
	reg_b = None
	for region in sim_set.region_set:
		count += 1
		if count == 5:
			reg_a = region

		if count == 6:
			reg_b = region
			break

	bins = [25,25,25]
	ranges = [0, 256, 0, 256, 0, 256]
	hist_a = sim_set.calculate_color_hist_of_region(reg_a, bins, ranges)

	# image_print_utils.print_bounding_box_region_and_seg_image(reg_a, pixel_class, sim_set, seg_image)
	# image_print_utils.print_bounding_box_region_and_seg_image(reg_b, pixel_class, sim_set, seg_image)

	s_val = sim_set.s_regions(reg_a, reg_b)

	# print("Simularity of region:{} and region:{} is: {}".format(reg_a, reg_b, s_val))
	# print hist_a.shape
	# image_print_utils.print_region_histogram(reg_a, im, sim_set)
	# image_print_utils.print_bounding_box_region_and_seg_image(reg, pixel_class, sim_set, seg_image)
