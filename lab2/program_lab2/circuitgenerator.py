import csv
import sys
import math
import random
import networkx as nx


class CircuitGenerator:
    def __init__(self, s, t, emf, nodes=10, type_='connected'):
        self.s = s
        self.t = t
        self.emf = emf
        self.nodes = int(nodes)
        self.type_ = type_
        self.edges = self.generate_circuit_edges_list()

    def generate_emf(self):
        with open('data/emf.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile,  delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('s', 't', 'e'))
            # sample values
            writer.writerow((self.s, self.t, self.emf))

    def generate_circuit(self):
        with open('data/circuit.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile,  delimiter=',', quotechar='\'', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(('u', 'v', 'r'))
            # sample values
            for row in self.edges:
                writer.writerow(row)

    def generate_circuit_edges_list(self):
        if self.type_ == 'connected':
            return self.generate_connected_edges_list()
        elif self.type_ == 'cubic':
            return self.generate_cubic_edges_list()
        elif self.type_ == 'bridge':
            return self.generate_bridge_edges_list()
        elif self.type_ == '2d':
            return self.generate_grid_2d_edges_list()

    def generate_grid_2d_edges_list(self):
        edges = []
        n = math.trunc(math.sqrt(self.nodes))
        g = nx.grid_2d_graph(n, n)
        for ((x, y), (z, w), d) in g.edges(data=True):
            edges.append((x*10 + y, z*10 + w, random.randint(1, 10)))
        return edges

    def generate_connected_edges_list(self):
        edges = []
        g = nx.erdos_renyi_graph(self.nodes, 0.5)
        for (u, v, d) in g.edges(data=True):
            edges.append((u, v, random.randint(1, 10)))
        return edges

    def generate_cubic_edges_list(self):
        edges = []
        g = nx.cubical_graph()
        for (u, v, d) in g.edges(data=True):
            edges.append((u, v, random.randint(1, 10)))
        return edges

    def generate_bridge_edges_list(self):
        return []