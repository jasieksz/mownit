import random
import math
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
import numpy as np

def generate_coords(n, i):
    return [[round(random.uniform(-n, n), 0), round(random.uniform(-n, n), 0)] for i in range(i)]


def distance(coord1, coord2):
    return round(math.sqrt(math.pow(coord1[0] - coord2[0], 2) + math.pow(coord1[1] - coord2[1], 2)), 4)


def create_distance_matrix(coords):
    n = len(coords)
    result = [[distance(coords[i], coords[j]) for i in range(n)] for j in range(n)]
    return result


#
# class Coordinates(object):
#     def __init__(self, n, elements):
#         # coordinates
#         self.n = n
#         self.iter = elements
#         self.coords = generate_coords(self.n, self.iter)
#         self.N = len(self.coords)
#         self.distance_matrix = create_distance_matrix(self.coords)




if __name__ == "__main__":
    size = 10
    elem = 5
    nodes = generate_coords(size, elem)
    edges = create_distance_matrix(nodes)
    print(edges)
    plt.plot(zip(*[nodes[edges[i % elem]] for i in range(elem+1)])[0], zip(*[nodes[edges[i % elem]] for i in range(elem)])[1],'xb-', );
    plt.show()
        

