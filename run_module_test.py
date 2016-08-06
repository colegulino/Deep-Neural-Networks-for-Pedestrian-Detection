# Here is where I run all of the packages module unit tests

set_of_test = ["image_disjoint_set.py", "image_graph.py", "simularity_set.py"]

for test in set_of_test:
	with open(test) as source_file:
		exec(source_file.read())
		
print("All Tests: COMPLETE")