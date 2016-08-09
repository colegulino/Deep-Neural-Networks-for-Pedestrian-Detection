# 
# Classes related to calculating a simularity set between segmentation regions
# 

import image_segmentation
import numpy as np

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
	def __init__(self, region_image, image, disjoint_set):
		self.region_image = region_image

		self.region_set = set()
		self.sim_set = set()
		self.image = image
		self.disjoint_set = disjoint_set

		self.height, self.width = region_image.shape
		self.image_size = self.height * self.width

		# Find the sets of neighboring regions
		for row in range(self.height):
			for col in range(self.width):
				current_region = self.region_image[row, col]
				if(col + 1 < self.width):
					neighbor_region = self.region_image[row, col + 1]
					self.add_set(current_region, neighbor_region)
				if(row + 1 < self.height):
					neighbor_region = self.region_image[row + 1, col]
					self.add_set(current_region, neighbor_region)
				if((col + 1 < self.width) and (row + 1 < self.height)):
					neighbor_region = self.region_image[row + 1, col + 1]
					self.add_set(current_region, neighbor_region)
				if((col + 1 < self.width) and (row > 0)):
					neighbor_region = self.region_image[row - 1, col + 1]
					self.add_set(current_region, neighbor_region)

		self.bounding_box = {}

		# Get the bounding boxes for each region
		for region in self.region_set:
			self.bounding_box[region] = self.create_bounding_box(region)

	# 
	# Returns the length of the regions set (number of regions currently available)
	# 
	# @return Number of regions in the simularity set
	# 
	def __len__(self):
		return len(self.region_set)

	# 
	# Allows you to add a set of regions to the simularity set
	# 
	# @param region_a One region in the set
	# @param region_b One region in the set
	# 
	def add_set(self, region_a, region_b):
		if region_a != region_b:
			if (region_b, region_a) not in self.sim_set:
				self.sim_set.add((region_a, region_b))
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
			self.sim_set.remove(s)
		if (s[1], s[0]) in self.sim_set:
			self.sim_set.remove((s[1], s[0]))

		# Determine if there are any more regions from set in the sim_set	
		if({item for item in self.sim_set if s[0] in item} == set()):
			print("Delete")
			self.region_set.remove(s[0])
		if({item for item in self.sim_set if s[1] in item} == set()):
			print("Delete")
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
	# A simularity function for regions based on size of the regions.
	# Encourages small regions to merge first
	# 
	# @param region_a One region to get a simularity for
	# @param region_b One region to get a simularity for
	# @return A real valued number that describes the simularity of the two regions
	# 
	def s_size(self, region_a, region_b):
		a = self.disjoint_set.find(region_a)
		b = self.disjoint_set.find(region_b)

		return 1 - ((a.size + b.size) / self.image_size)

if __name__ == "__main__":
	# Show the image before segmentation
	
	print("Testing simularity set")

	region_image = np.array([[1, 2, 3], [4, 5, 6], [4, 4, 6]])
	ss = simularity_set(region_image)

	region_set = ss.regions()
	sim_set = ss.get_sim_set()
	ss.remove_set((1,2))
	ss.remove_set((1,4))
	ss.remove_set((1,5))

	print("Number of regions (should be 5): {}".format(len(ss)))
	print("Number of sets (should be 9): {}".format(len(sim_set)))
