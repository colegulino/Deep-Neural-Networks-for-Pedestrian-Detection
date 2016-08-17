# 
# Classes related to calculating a simularity set between segmentation regions
# 

import image_segmentation
import histogram_utils
import image_print_utils

from PIL import Image
import numpy as np
import cv2
import operator

class bounding_box:
	def __init__(self, region, row_start, col_start, height, width):
		self.region = region
		self.row_start = row_start
		self.col_start = col_start
		self.height = height
		self.width = width

		self.x0 = self.col_start
		self.y0 = self.row_start

	def __len__(self):
		return self.height * self.width

class simularity_set:
	def __init__(self, region_image, image, disjoint_set, seg_image):
		self.region_image = region_image

		self.region_set = set()
		self.sim_set = {}
		self.image = image
		self.disjoint_set = disjoint_set
		self.seg_image = np.array(seg_image)

		self.height, self.width = region_image.shape
		self.image_size = self.height * self.width

		# Find the sets of neighboring regions
		self.find_neighboring_regions()

		# Get the bounding boxes for each region
		self.bounding_box = {}
		for region in self.region_set:
			self.bounding_box[region] = self.create_bounding_box(region)

		self.get_region_simularities()

	# 
	# Returns the length of the regions set (number of regions currently available)
	# 
	# @return Number of regions in the simularity set
	# 
	def __len__(self):
		return len(self.region_set)

	# 
	# Function that finds all the neighboring regions in the image
	# 
	def find_neighboring_regions(self):
		for row in range(self.height):
			for col in range(self.width):
				current_region = self.region_image[row, col]
				if col + 1 < self.width:
					neighbor_region = self.region_image[row, col + 1]
					self.add_set(current_region, neighbor_region)
				if col - 1 > 0:
					neighbor_region = self.region_image[row, col - 1]
					self.add_set(current_region, neighbor_region)
				if row + 1 < self.height:
					neighbor_region = self.region_image[row + 1, col]
					self.add_set(current_region, neighbor_region)
				if row - 1 > 0:
					neighbor_region = self.region_image[row - 1, col]
					self.add_set(current_region, neighbor_region)
				if col - 1 > 0 and row - 1 > 0:
					neighbor_region = self.region_image[row - 1, col - 1]
					self.add_set(current_region, neighbor_region)
				if col - 1 > 0 and row + 1 < self.height:
					neighbor_region = self.region_image[row + 1, col - 1]
					self.add_set(current_region, neighbor_region)
				if col + 1 < self.width and row - 1 > 0:
					neighbor_region = self.region_image[row - 1, col + 1]
					self.add_set(current_region, neighbor_region)
				if col + 1 < self.width and row + 1 < self.height:
					neighbor_region = self.region_image[row + 1, col + 1]
					self.add_set(current_region, neighbor_region)

	# 
	# Allows you to add a set of regions to the simularity set
	# 
	# @param region_a One region in the set
	# @param region_b One region in the set
	# 
	def add_set(self, region_a, region_b):
		if region_a != region_b:
			if (region_a, region_b) not in self.sim_set.keys() and \
			   (region_b, region_a) not in self.sim_set.keys():
				self.sim_set[(region_a, region_b)] = 0
			self.region_set.add(region_a)
			self.region_set.add(region_b)

	# 
	# Allows you to get at the set of adjacent regions
	# 
	# @return self.sim_set
	# 
	def get_sim_set(self):
		return self.sim_set

	# 
	# Allows you to get the set of regions present in the sim set
	# 
	# @return Set of unique regions
	# 
	def get_regions(self):
		return self.region_set

	# 
	# Remove a set from the simularity set
	# 
	# @param s Set to remove as tuple (region_a, region_b)
	# 
	def remove_set(self, s):
		if s in self.sim_set:
			self.sim_set.pop(s, None)
		if (s[1], s[0]) in self.sim_set:
			self.sim_set.pop((s[1], s[0]), None)

		# Determine if there are any more regions from set in the sim_set	
		if({key for key, value in self.sim_set.items() if s[0] in key} == set()):
			self.region_set.remove(s[0])
		if({key for key, value in self.sim_set.items() if s[1] in key} == set()):
			self.region_set.remove(s[1])

	# 
	# Function that returns a list of indices (row, col) where a region lies on the image
	# 
	# @param region Region to get the indices for
	# @return A list of indices that specify where a region is in an image
	# 
	def get_indices_of_region(self, region):
		return np.transpose((self.region_image == region).nonzero())	

	# 
	# Creates and returns a bounding box around the region specified
	# 
	# @param region Region to create the bounding box around
	# @return Bounding box around the region
	# 
	def create_bounding_box(self, region):
		indices = self.get_indices_of_region(region)
		row_start = min(indices[:, 0])
		col_start = min(indices[:, 1])
		height = max(indices[:, 0]) - row_start
		width = max(indices[:, 1]) - col_start

		return bounding_box(region, row_start, col_start, height, width)

	# 
	# Returns the bounding box that combines both regions
	# 
	# @param box_a Bounding box around region a
	# @param box_b Bounding box around region b
	# @return A bounding box around region a and region b
	# 
	def combine_bounding_boxes(self, box_a, box_b):
		row_start = min(box_a.row_start, box_b.row_start)
		col_start = min(box_a.col_start, box_b.col_start)

		max_row_a = box_a.row_start + box_a.height
		max_col_a = box_a.col_start + box_a.width
		max_row_b = box_b.row_start + box_b.height
		max_col_b = box_b.row_start + box_b.width

		height = max(max_row_a, max_row_b) - row_start
		width = max(max_col_a, max_col_b) - col_start

		return bounding_box((box_a.region, box_b.region), row_start, col_start, height, width)

	# 
	# Gets a mask the shape of the image with 1s where the region exists
	# 
	# @param region Region to get the mask for
	# @param dtype Datatype you want the region mask to be
	# @return a mask based on where region is in the image
	# 
	def get_region_mask(self, region, dtype):
		return (self.region_image == region).astype(dtype)

	# 
	# Calculates a color histogram based on the region
	# 
	# @param region Region to get the color histogram for
	# @return A color histogram based on the region with 25 bins per channel
	# 
	def calculate_color_hist_of_region(self, region, bins=[25,25,25], ranges=[0,256,0,256,0,256]):
		mask = self.get_region_mask(region, dtype=np.uint8)
		channels = [0, 1, 2]
		return histogram_utils.get_normalized_histogram(self.image, channels, bins, ranges, mask)

	# 
	# A simularity function for regions based on size of the regions.
	# Encourages small regions to merge first
	# 
	# @param region_a One region to get a simularity for
	# @param region_b One region to get a simularity for
	# @return A real valued number that describes the simularity of the two regions
	# 
	def s_size(self, region_a, region_b):
		a = self.disjoint_set.get(region_a)
		b = self.disjoint_set.get(region_b)

		return 1 - ((float(a.size + b.size) / self.image_size))

	# 
	# A simularity function for regions based on the fill of the regions
	# Ensures that combined shapes make a natural shape
	# 
	# @param region_a One region to get a simularity for
	# @param region_b One region to get a simularity for
	# @return A real valued number that describes the simularity of the two regions
	# 
	def s_fill(self, region_a, region_b):
		a = self.disjoint_set.get(region_a)
		b = self.disjoint_set.get(region_b)

		bbox_a = self.create_bounding_box(a.parent)
		bbox_b = self.create_bounding_box(b.parent)
		bbox_ab = self.combine_bounding_boxes(bbox_a, bbox_b)

		return 1 - (float(len(bbox_ab) - a.size - b.size) / self.image_size)

	# 
	# A simularity function for regions based on color simularity
	# 
	# @param region_a One region to get a simularity for
	# @param region_b One region to get a simularity for
	# @return A real valued number that describes the simularity of the two regions
	# 
	def s_color(self, region_a, region_b, method_name='intersection'):
		a = self.disjoint_set.get(region_a)
		b = self.disjoint_set.get(region_b)

		hist_a = self.calculate_color_hist_of_region(a.parent)
		hist_b = self.calculate_color_hist_of_region(b.parent)

		return histogram_utils.normalized_histogram_intersection(hist_a, a.size, hist_b, b.size)

	# 
	# A simularity for regions based on texture using SIFT features
	# 
	# @param region_a One region to get simularity for
	# @param region_b One region to get simularity for
	# @return A real valued number that describes the simularity of the two regions
	# 
	def s_texture(self, region_a, region_b, method_name='intersection'):
		a = self.disjoint_set.find(region_a)
		b = self.disjoint_set.find(region_b)

		mask_a = self.get_region_mask(a, dtype=int)
		mask_b = self.get_region_mask(b, dtype=int)

		hist_a = histogram_utils.get_sift_features(self.image, mask=mask_a)
		hist_b = histogram_utils.get_sift_features(self.image, mask=mask_b)

		return histogram_utils.compare_histograms(hist_a, hist_b, method_name)

	# 
	# Find the simularity between two regions
	# 
	# @param region_a One region to find a simularity for
	# @param region_b One region to find a simularity for
	# @param a A tuple that is a binary mask on which parameters to include
	# @return A real valued number that describes the simularity of the two regions
	# 
	def s_regions(self, region_a, region_b, a=(1,1,1,1)):
		if len(a) != 4:
			raise ValueError('Size of a ({}) should be: {}'.format(len(a),4))

		return a[0] * self.s_size(region_a, region_b) + \
			   a[1] * self.s_fill(region_a, region_b) + \
			   a[2] * self.s_color(region_a, region_b) + \
			   a[3] * self.s_texture(region_a, region_b)

	# 
	# Find and record the simularities between all of the regions in the set
	# 
	def get_region_simularities(self):
		for s in self.sim_set.keys():
			simul = self.s_regions(s[0], s[1])
			self.sim_set[s] = simul

	# 
	# Function to merge two regions together
	# 
	# @param region_a One of the regions to merge
	# @param region_b One of the regions to merge
	# @return Bounding box shape for the new region
	# 
	def merge_regions(self, region_a, region_b):
		# Update the disjoint set by joining the two regions
		self.disjoint_set.union(region_a, region_b)
		parent_region = self.disjoint_set.find(region_a)
		if parent_region == region_b:
			region_a, region_b = region_b, region_a 

		# Get all sets that contain region_a or region_b
		neighbor_set = {key for key, value in self.sim_set.items() \
						if region_a in key or region_b in key}

		# Replace all the sets of neighbor regions that contain region_a or region_b
		# with a set including the new parent region and the other neighbor 
		for neighbor in neighbor_set:
			del self.sim_set[neighbor]
			if not (region_a in neighbor and region_b in neighbor):
				reg1, reg2 = neighbor
				if reg1 == region_a or reg1 == region_b:
					self.sim_set[(region_a, reg2)] = self.s_regions(region_a, reg2)
				else:
					self.sim_set[(region_a, reg1)] = self.s_regions(region_a, reg1)

		# Update the segmented image and region_image
		indices_a = self.get_indices_of_region(region_a)
		row = indices_a[0,0]
		col = indices_a[0,1]
		color = self.seg_image[row, col, :]

		indices_b = self.get_indices_of_region(region_b)
		for ind in indices_b:
			row = ind[0]
			col = ind[1]
			self.seg_image[row, col, 0] = color[0]	
			self.seg_image[row, col, 1] = color[1]	
			self.seg_image[row, col, 2] = color[2]

			self.region_image[row, col] = region_a

		# Update the bounding boxes
		del self.bounding_box[region_b]
		self.bounding_box[region_a] = self.create_bounding_box(region_a)

		return self.bounding_box[region_a]
	# 
	# Function that returns the two regions that are the most similiar according to the function
	# s_regions
	# 
	# @return A pair of regions that denote the two regions with the highest similarity
	# 
	def get_most_similar_regions(self):
		regions, simul = max(self.sim_set.items(), key=operator.itemgetter(1))
		return regions

if __name__ == "__main__":
	# Show the image before segmentation
	
	print("Testing simularity set")

	region_image = np.array([[1, 2, 3], [4, 5, 6], [4, 4, 6]])
	ss = simularity_set(region_image, None, None)

	region_set = ss.get_regions()
	sim_set = ss.get_sim_set()
	ss.remove_set((1,2))
	ss.remove_set((1,4))
	ss.remove_set((1,5))

	print("Number of regions (should be 5): {}".format(len(ss)))
	print("Number of sets (should be 9): {}".format(len(sim_set)))
