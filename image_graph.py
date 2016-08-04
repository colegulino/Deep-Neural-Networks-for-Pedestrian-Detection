# 
# Classes for creating an image graph
# 

class edge:
	def __init__(self, e, weight):
		self.a = e[0]
		self.b = e[1]
		self.e = e
		self.weight = weight

	def __str__(self):
		return "a={}, b={}, e={}, weight={}".format(self.a,self.b,self.e,self.weight)

class image_graph:
	def __init__(self, edges = None):
		if edges == None:
			self.edges = {}
		else:
			self.edges = edges

		self.edge_list = None

	def __str__(self):
		return "Graph with {} Number of Edges".format(len(self.edges))

	def __len__(self):
		return len(self.edges)

	# 
	# Add an edge to the graph
	# 
	def add_edge(self, edge_class=None, a=None, b=None, e=None, weight=None):
		if(edge_class != None):
			self.edges[edge_class.e] = edge_class
		elif(e != None and weight != None):
			self.edges[e] = edge(e, weight)
		elif(a != None and b != None and weight != None):
			self.edges[(a,b)] = edge((a,b), weight)
		else:
			print("Not enough arguments: edge={}, a={}, b={}, e={}, weight={}".format(edge,a,b,e,weight))

	# 
	# Reuturn a sorted list of the edges by weight (ascending)
	# 
	def get_sorted_edge_list(self):
		self.edge_list = [value for key, value in self.edges.items()]
		return sorted(self.edge_list, key=lambda edge: edge.weight, reverse=True)

	# 
	# Return the edge dict
	# 
	def edges(self):
		return self.edges

if __name__ == "__main__":
	# Test the image graph class

	print("Running test for image_graph class")

	e1 = edge((1,2), 10)
	e2 = edge((2,3), 20)
	e3 = edge((3,4), 30)

	gr = image_graph()

	gr.add_edge(edge_class=e1)
	gr.add_edge(a=e2.a, b=e2.b, weight=e2.weight)
	gr.add_edge(e=(e3.a, e3.b), weight=e3.weight)

	edge_list = gr.get_sorted_edge_list()

	print("Printing Sorted Edge List")
	print(edge_list[0])
	print(edge_list[1])
	print(edge_list[2])
	print(gr)
	print("Length of graph: {}".format(len(gr)))
	print("Image Graph Tests complete!")
