# Random int
from random import randint

# Import my graph class
import image_graph
# Import my image disjoint set
import image_disjoint_set

# Import numpy and scipy
from scipy.ndimage import filters
import numpy as np

# Import OpenCV and image
import cv2
from PIL import Image
from pylab import *

# 
# Function that returns the index of the vectorized version of the image index from the current row and column
# 
# @param row Row of current index of the image
# @param col Column of current index of image
# @param shape Shape of image
# @return Vectorized image index
# 
def get_flattened_index(row, col, shape):
	height, width, channels = shape
	return row * width + col

# 
# A dissimiliarity function for two pixels using L2 distance between pixels
# 
# @param image The image where the pixels exist
# @param index_a The shape (row, col) of the first pixel
# @param index_b The shape (row, col) of the second pixel
# @return A measure of the dissimiliarity of the two pixels
# 
def diff_L2(image, index_a, index_b):
	return ((image[index_a[0], index_a[1], 0] - image[index_b[0], index_b[1], 0])**2 + \
		    (image[index_a[0], index_a[1], 1] - image[index_b[0], index_b[1], 1])**2 + \
		    (image[index_a[0], index_a[1], 2] - image[index_b[0], index_b[1], 2])**2)**0.5

#
# A dissimilarity function between image pixels using exponential
# 
# @param image The image where the pixels exist
# @param index_a The index (row, col) of the first pixel
# @param index_b The index (row, col) of the second pixel
# @return A measure of the dissimilarity of the two pixels
# 
kappa = 2
sig = 1e2
def exp_diff(image, index_a, index_b):
	val_a = image[index_a[0], index_a[1], :]
	val_b = image[index_b[0], index_b[1], :]

	return kappa * exp(-1.0 * sum((val_a - val_b)**2) / sig)

# 
# Function for generating the graph to be cut
# 
# @param image Image to generate a graph from
# @param weight_fxn Function that generates the weights for the graph. The function should take in three parameters:
#        1. The image (row, cols, channels) 2. Index of one pixel (row, col) 3. Index of the other pixel (row, col)
# @return A graph with weights generated from weight_fxn
# 
def generate_graph(image, weight_fxn):
	height, width, channels = image.shape
	graph = image_graph.image_graph()

	for row in range(height):
		for col in range(width):
			current_node = get_flattened_index(row, col, image.shape)
			current_index = (row, col)

			if(col + 1 < width):
				right_neighbor = get_flattened_index(row, col+1, image.shape)
				right_index = (row, col+1)
				graph.add_edge(e=(current_node, right_neighbor), 
					            weight=weight_fxn(image, current_index, right_index))
			if(row + 1 < height):
				bottom_neighbor = get_flattened_index(row+1, col, image.shape)
				bottom_index = (row+1, col)
				graph.add_edge(e=(current_node, bottom_neighbor), 
					            weight=weight_fxn(image, current_index, right_index))
			if((col < width - 1) and (row < height - 1)):
				bottom_right_neighbor = get_flattened_index(row+1, col+1, image.shape)
				bottom_right_index = (row+1, col+1)
				graph.add_edge(e=(current_node, bottom_right_neighbor), \
					            weight=weight_fxn(image, current_index, bottom_right_index))
			if((col < width - 1) and (row > 0)):
				top_right_neighbor = get_flattened_index(row-1, col+1, image.shape)
				top_right_index = (row-1, col+1)
				graph.add_edge(e=(current_node, top_right_neighbor), 
					            weight=weight_fxn(image, current_index, top_right_index))

	return graph

# 
# Function that segments an image
# 
# @param image The image to segment
# @param sigma Gaussian filter sigma
# @param th Threshold 
# @param min_size Minimum component size
# @return A segmented image and the number of current connected components
# 
def segment_image(image, sigma, th, min_size):
	height, width, channels = image.shape
	num_vertices = height * width

	# Normalize the images
	image = image.reshape((-1, 3))
	image = image / np.linalg.norm(image)
	image = image.reshape((height, width, channels))

	print("Height: {}, Width: {}, Channels: {}".format(height, width, channels))

	# Smooth out the RGB channels in order to reduce noise in the image
	for i in range(3):
		image[:,:,i] = filters.gaussian_filter(image[:,:,1], sigma)

	print("Generating graph...")
	graph = generate_graph(image, exp_diff)
	print("Finished generated graph.")

	print("Generating Disjoint Set from Graph...")
	disjoint_set = image_disjoint_set.image_disjoint_set(num_elements=num_vertices)

	def threshold(size, th):
		return th / size

	# Initialize thresholds for each element
	thresholds = {}
	for i in range(num_vertices):
		thresholds[i] = threshold(1,th)

	sorted_edges = graph.get_sorted_edge_list()
	for i in range(len(graph)):
		edge = sorted_edges[i]
		a = disjoint_set.find(edge.a)
		b = disjoint_set.find(edge.b)
		if a != b:
			if edge.weight <= thresholds[a] and edge.weight <= thresholds[b]:
				disjoint_set.union(a,b)
				a_parent = disjoint_set.find(a)
				thresholds[a_parent] = edge.weight + threshold(disjoint_set.get_set_size(a_parent), th)

	# Merge very small components
	for i in range(len(graph)):
		edge = sorted_edges[i]
		a = disjoint_set.find(edge.a)
		b = disjoint_set.find(edge.b)

		if(a != b and (disjoint_set.get_set_size(a) < min_size or disjoint_set.get_set_size(b) < min_size)):
			disjoint_set.union(a,b)
	
	print("Finished generating Disjoint set.")
	print("Number of sets generated: {}".format(disjoint_set.num_sets))
	colors = {}
	for i in range(width * height):
		colors[i] = (randint(0,255), randint(0,255), randint(0,255))

	print("Generating segmented image")
	seg_image = np.ndarray(shape=image.shape)
	pixel_class = np.ndarray(shape=(height, width))

	for row in range(height):
		for col in range(width):
			index = get_flattened_index(row, col, image.shape)
			parent = disjoint_set.find(index)
			color = np.uint8(colors[parent])
			seg_image[row, col, 0] = color[0]
			seg_image[row, col, 1] = color[1]
			seg_image[row, col, 2] = color[2]
			pixel_class[row, col] = parent

	seg_image = np.uint8(seg_image)
	return Image.fromarray(seg_image), pixel_class, disjoint_set

if __name__ == "__main__":
	# Show the image before segmentation
	
	image_name = "large_pedestrian.jpg"
	print("Testing image segmentation aggorithm on {}".format(image_name))
	im = Image.open(image_name)

	# Parameters
	sigma = 0.5
	min_size = 50
	th = 500 # Large th results in larget components

	seg_image, pixel_class = segment_image(im, sigma, th, min_size)

	fig = figure()
	fig.add_subplot(2, 1, 1)
	imshow(im)
	fig.add_subplot(2, 1, 2)
	imshow(seg_image)
	show()