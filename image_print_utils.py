import image_segmentation
import simularity_set

from PIL import Image
from pylab import *
from matplotlib.patches import Rectangle

import numpy as np

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