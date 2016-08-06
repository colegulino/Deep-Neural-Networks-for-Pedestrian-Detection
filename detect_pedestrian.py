# Here is a general python script for running a prototype of the algorithm described in
# Effecient Graph-Based Image Segmentation by Felzenzwalb and Huttenlocher
# Link: https://cs.brown.edu/~pff/papers/seg-ijcv.pdf
# Source Code Link: http://cs.brown.edu/~pff/segment/ 

import image_segmentation
import simularity_set

from PIL import Image
from pylab import *

import numpy as np

if __name__ == "__main__":
	# Show the image before segmentation
	
	image_name = "treeAndHorse.jpg"
	# image_name = "large_pedestrian.jpg"
	im = Image.open(image_name)

	# Parameters
	sigma = 0.5
	min_size = 100
	th = 5000 # Large th results in larget components

	seg_image, pixel_class, disjoint_set = image_segmentation.segment_image(im, sigma, th, min_size)

	print("Get simularity set")
	sim_set = simularity_set.simularity_set()

	print(pixel_class)

	fig = figure()
	fig.add_subplot(2, 1, 1)
	imshow(im)
	fig.add_subplot(2, 1, 2)
	imshow(seg_image)
	show()