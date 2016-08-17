#!/usr/bin/python

# 
# Interface code to generate regions of an image that can be used for object detection
# 

import image_segmentation
import simularity_set
import image_print_utils
import histogram_utils

from PIL import Image

import sys, getopt
import yaml

import numpy as np

image_name = ''
output_file = ''
parameter_file = ''

try:
	opts, args = getopt.getopt(sys.argv[1:], "hi:o:p:")
except getopt.GetoptError:
	print('get_candidate_regions.py -i <input_image> -o <output_file> -p <param_file>')
	print('get_candidate_regions.py -h for more help')
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print('--------------------------------------------------------------------------------------')
		print("This function generates bounding box values for candidate regions that can be used for")
		print("object detection candidate regions.")
		print("--------------------------------------------------------------------------------------")
		print('get_candidate_regions.py -i <input_image> -o <output_file> -p <param_file>')
		print('-i <input_image>')
		print('Image to generate the bounding box candidates for. Must be an image format')
		print("")
		print('-o <output_file>')
		print('Output file that contains the bounding box locations of each of the regions.')
		print('Format of the file: each line contains one region with format: x0 y0 width height.')
		print("")
		print('-p <param_file>')
		print('.yml file that specifies the parameters for the segmentation')
		print('Parameters:')
		print('	sigma = Gaussian filter sigma')
		print('	min_size = Minimum size for a region allowable')
		print('	th = Threshold for region segmentation. The larger this is, the larger the initial regions are.')
		print("--------------------------------------------------------------------------------------")
		sys.exit()
	elif opt == '-i':
		image_name = arg
	elif opt == '-o':
		output_file = arg
	elif opt == '-p':
		parameter_file = arg

if image_name == "":
	print('Must specify an image name. Use -h for more information.')
	sys.exit(2)
print("Input Image: {}".format(image_name))

if output_file == "":
	print('Using default output file name: out.txt')
	output_file = 'out.txt'
print("Output Name: {}".format(output_file))

if parameter_file == "":
	sigma = 0.5
	min_size = 100
	th = 3000
	print('Using default parameter values: sigma={}, min_size={}, th={}'.format(sigma,min_size,th))
else:
	print("Parameter File: {}".format(parameter_file))
	with open(parameter_file, 'r') as yml_file:
		params = yaml.load(yml_file)
	sigma = params["segmentation_params"]["sigma"]
	min_size = params["segmentation_params"]["min_size"]
	th = params["segmentation_params"]["th"]
	print('Using Found Parameters: sigma={}, min_size={}, th={}'.format(sigma,min_size,th))

try:
	im = Image.open(image_name)
except IOError:
	print("Image name {} is not a valid image format.".format(image_name))
	sys.exit(2)	
im = im.convert('RGB')
im = np.array(im)

seg_image, pixel_class, disjoint_set = image_segmentation.segment_image(im, sigma, th, min_size)

print("Getting simularity set...")
sim_set = simularity_set.simularity_set(region_image=pixel_class, image=im, \
										disjoint_set=disjoint_set, seg_image=seg_image)
print("Got simularity set.")

print('Getting the region information...')
bounding_boxes = set()
# Get all of the initial regions
for region, bbox in sim_set.bounding_box.items():
	bounding_boxes.add(bbox)

# Get the rest of the regions as you combine the best ones
while sim_set.disjoint_set.num_sets > 1:
	reg_a, reg_b = sim_set.get_most_similar_regions()
	bbox = sim_set.merge_regions(reg_a, reg_b)
	bounding_boxes.add(bbox)
print('Got all the region information.')

print("Storing the region information...")
with open(output_file, 'w') as file:
	for bbox in bounding_boxes:
		file.write('{} {} {} {}\n'.format(bbox.x0, bbox.y0, bbox.width, bbox.height))
print("Stored all the region infromation in file: {}".format(output_file))