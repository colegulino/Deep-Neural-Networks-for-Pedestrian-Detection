# 
# Classes related to a disjoint set forests created using union-by-rank
# Based on implementation found:  https://en.wikipedia.org/wiki/Disjoint-set_data_structure
# 

class image_disjoint_set_element:
	def __init__(self, rank, parent, size):
		self.rank = rank
		self.parent = parent
		self.size = size

	def __str__(self):
		return "Disjoint Set Element: rank: {}, parent: {}, size: {}".format(self.rank, self.parent, self.size)	

class image_disjoint_set:
	def __init__(self, num_elements=0):
		if(num_elements <= 0):
			raise ValueError('Number of elements must be >= 0. Num elements provided is: {}'.format(num_elemets))

		self.num_sets = num_elements
		self.elements = {}

		for i in range(self.num_sets):
			self.elements[i] = image_disjoint_set_element(rank=0, parent=i, size=1)

	def __str__(self):
		return "Disjoint set with {} number of elements".format(self.size)

	def __len__(self):
		return len(self.elements)

	# 
	# Get the size of the set an element is in
	# 
	# @param x The element to get the size of
	# 
	def get_set_size(self, x):
		return self.elements[x].size

	# 
	# Run find recursively on the index in the set
	# Utilizes path flattening so that all those in the same disjoint set have the same parent
	# This makes the tree flatter and easier to search through
	# 
	# @param x The index to search for
	# @return The parent index of the element (which subset it belongs to)
	#  
	def find(self, x):
		# parent = self.elements[x].parent
		# if parent != x:
		# 	parent = self.find(parent)
		# return parent	
		y = x
		while y != self.elements[y].parent:
			y = self.elements[y].parent
		self.elements[x].parent = y
		return y

	# 
	# Combine two elements into the same set
	# 
	# @param x One element to be combined
	# @param y The othe element to be combined
	#
	def union(self, x, y):
		x_root = self.elements[self.find(x)]
		y_root = self.elements[self.find(y)]

		# If they share the same root (same subset) no need to form union
		if x_root.parent == y_root.parent:
			return

		# If x and y are in the same set, merge them
		if x_root.rank < y_root.rank:
			x_root.parent = y_root.parent
			y_root.size += x_root.size
		elif y_root.rank < x_root.rank:
			y_root.parent = x_root.parent
			x_root.size += y_root.size
		else:
			y_root.parent = x_root.parent
			x_root.size += y_root.size
			x_root.rank = x_root.rank + 1

		self.num_sets -= 1

		# print("Num sets: {}".format(self.num_sets))

if __name__ == "__main__":
	# Test the disjoint set class
	
	print("Running tests for image_disjoint_set")

	ds = image_disjoint_set(10)

	for i in range(len(ds)):
		a = ds.find(i)

	print("Number of sets (should be 10): {}".format(ds.num_sets))

	ds.union(1,2)

	print("Size of set 1 (should be 2): {}".format(ds.get_set_size(1)))
	print("Size of set 2 (should be 1): {}".format(ds.get_set_size(2)))

	print("Number of sets (should be 9): {}".format(ds.num_sets))

	print("Disjoint Set Tests Completed")
