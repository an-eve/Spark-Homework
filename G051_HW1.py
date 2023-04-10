from pyspark import SparkContext, SparkConf
import sys
import os
import random as rand
from collections import defaultdict
import statistics
import time

def CountTriangles1(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    yield triangle_count
    
def CountTriangles(edges):
    # Create a defaultdict to store the neighbors of each vertex
    neighbors = defaultdict(set)
    for edge in edges:
        u, v = edge
        neighbors[u].add(v)
        neighbors[v].add(u)

    # Initialize the triangle count to zero
    triangle_count = 0

    # Iterate over each vertex in the graph.
    # To avoid duplicates, we count a triangle <u, v, w> only if u<v<w
    for u in neighbors:
        # Iterate over each pair of neighbors of u
        for v in neighbors[u]:
            if v > u:
                for w in neighbors[v]:
                    # If w is also a neighbor of u, then we have a triangle
                    if w > v and w in neighbors[u]:
                        triangle_count += 1
    # Return the total number of triangles in the graph
    return triangle_count




def MR_ApproxTCwithNodeColors(rawData, C):
	p=8191
	a= rand.randint(1,p-1)
	b= rand.randint(0,p-1)

	def hashfColor(vertex):
		return (((a*vertex)+b) % p) % C
	
	triangleCountMap= rawData.map(lambda x: MapR1(x, hashfColor)).filter(lambda x: x!= None).groupByKey().map(lambda x : (x[0], list(x[1])))
	triangleCountMap2= triangleCountMap.map(lambda x: CountTriangles(x[1])).reduce(lambda x,y: x + y)
    
	return C**2 * triangleCountMap2

def MapR1(row, hashfColor):  #row("0,1")
	l1= list(map(int,row.split(",")))
	l_hashed=list(map(hashfColor,l1))
	if l_hashed[0]== l_hashed[1]:
		return (l_hashed[0],tuple(l1))
	

def MR_ApproxTCwithSparkPartitions(edges, C):
	# Round 1: count the number of triangles in each partition
	t_per_partition = edges.mapPartitions(CountTriangles1).collect()
  
	# Round 2: compute the total number of triangles
	t_final = (C ** 2) * sum(t_per_partition)

	return t_final




def main():

	# CHECKING NUMBER OF CMD LINE PARAMTERS
	assert len(sys.argv) == 4, "Usage: python WordCountExample.py <C> <R> <file_name>"

	# SPARK SETUP
	conf = SparkConf().setAppName('G051HW1')
	sc = SparkContext(conf=conf)

	# INPUT READING

	# 1. Read C and R 
	C = sys.argv[1]
	assert C.isdigit(), "K must be an integer"
	C = int(C)
	R = sys.argv[2]
	assert R.isdigit(), "K must be an integer"
	R = int(R)

	# 2. Read input file and subdivide it into K random partitions
	data_path = sys.argv[3]
	assert os.path.isfile(data_path), "File or folder not found"
	rawData = sc.textFile(data_path,).cache()

	time0 = time.time()
	final_values=[]
	for i in range(R):
		final_values.append(MR_ApproxTCwithNodeColors(rawData,C))
    
	time1 = time.time()

	
	mean_values= statistics.median(final_values)
	print("RISULTATO", mean_values)
        
	
	edges = rawData.map(lambda x: list(map(int, x.split(',')))).repartition(C).cache()
	time22=time.time()
	print("RISULTATO 1", MR_ApproxTCwithSparkPartitions(edges, C))

	time2 = time.time()
    
	print("time ", (time1-time0)/R*1000,(time2-time22)*1000, sep=" ")


if __name__ == "__main__":
	main()

