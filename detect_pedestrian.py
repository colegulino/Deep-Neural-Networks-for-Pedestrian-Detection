# Here is a general python script for running a prototype of the algorithm described in
# Effecient Graph-Based Image Segmentation by Felzenzwalb and Huttenlocher
# Link: https://cs.brown.edu/~pff/papers/seg-ijcv.pdf
# Source Code Link: http://cs.brown.edu/~pff/segment/ 

import image_segmentation
import simularity_set

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

	bbox = None

	for region in sim_set.region_set:
		bbox = sim_set.create_bounding_box(region)
		break

	fig = figure()
	# fig.add_subplot(2, 1, 1)
	# imshow(im)
	# fig.add_subplot(2, 1, 2)
	rect = Rectangle([bbox.x0, bbox.y0], bbox.width, bbox.height, color=[0,0,0], fill=False)
	fig.patches.append(rect)
	imshow(seg_image)
	show()