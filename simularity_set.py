# 
# Classes related to calculating a simularity set between segmentation regions
# 

import image_segmentation
import numpy as np

class simularity_set:
	def __init__(self, class_image):
		self.class_image = class_image

		self.class_set = set()
		self.sim_set = set()

		height, width = class_image.shape

		# Find the sets of regions
		for row in range(height):
			for col in range(width):
				current_class = self.class_image[row, col]
				if(col + 1 < width):
					neighbor_class = self.class_image[row, col + 1]
					self.add_set(current_class, neighbor_class)
				if(row + 1 < height):
					neighbor_class = self.class_image[row + 1, col]
					self.add_set(current_class, neighbor_class)
				if((col + 1 < width) and (row + 1 < height)):
					neighbor_class = self.class_image[row + 1, col + 1]
					self.add_set(current_class, neighbor_class)
				if((col + 1 < width) and (row > 0)):
					neighbor_class = self.class_image[row - 1, col + 1]
					self.add_set(current_class, neighbor_class)

	# 
	# Returns the length of the class set (number of classes currently available)
	# 
	# @return Number of classes in the simularity set
	# 
	def __len__(self):
		return len(self.class_set)

	# 
	# Allows you to add a set of classes to the simularity set
	# 
	# @param class_a One class in the set
	# @param class_b One class in the set
	# 
	def add_set(self, class_a, class_b):
		if class_a != class_b:
			if (class_b, class_a) not in self.sim_set:
				self.sim_set.add((class_a, class_b))

			self.class_set.add(class_a)
			self.class_set.add(class_b)

	# 
	# Allows you to get at the set of adjacent classes
	# 
	# @return self.sim_set
	# 
	def get_sim_set(self):
		return self.sim_set

	# 
	# Allows you to get the set of classes present in the sim set
	# 
	# @return Set of unique classes
	# 
	def get_classes(self):
		return self.class_set

	# 
	# Remove a set from the simularity set
	# 
	# @param s Set to remove as tuple (class_a, class_b)
	# 
	def remove_set(self, s):
		if s in self.sim_set:
			self.sim_set.remove(s)
		if (s[1], s[0]) in self.sim_set:
			self.sim_set.remove((s[1], s[0]))

		# Determine if there are any more classes from set in the sim_set
		if({item for item in self.sim_set if s[0] in item} == set()):
			print("Delete")
			self.class_set.remove(s[0])
		if({item for item in self.sim_set if s[1] in item} == set()):
			print("Delete")
			self.class_set.remove(s[1])

if __name__ == "__main__":
	# Show the image before segmentation
	
	print("Testing simularity set")

	class_image = np.array([[1, 2, 3], [4, 5, 6], [4, 4, 6]])
	ss = simularity_set(class_image)

	class_set = ss.get_classes()
	sim_set = ss.get_sim_set()
	ss.remove_set((1,2))
	ss.remove_set((1,4))
	ss.remove_set((1,5))

	print("Number of classes (should be 5): {}".format(len(ss)))
	print("Number of sets (should be 9): {}".format(len(sim_set)))
