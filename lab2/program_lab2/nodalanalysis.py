import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


class Solver(object):
    def __init__(self):
        self.path = "./in.txt"
        self.G = nx.DiGraph()
        self.sem = self.read_sem()

    def read_sem(self):
        f = open(self.path, 'r')
        line = f.readline()
        sem_line = line.split(" ")  # SEM
        sem_tup = (int(sem_line[0]), int(sem_line[1]), float(sem_line[2]))
        f.close()
        return sem_tup

    def load(self):
        edge_iter = 0
        f = open(self.path, 'r')
        line = f.readline()
        line = f.readline()
        while line and line not in ["", "\n"]:
            split_line = line.split(" ")
            u = int(split_line[0])
            v = int(split_line[1])
            w = float(split_line[2])
            self.G.add_weighted_edges_from([(u, v, {'r': w, 'i': 0, 'num': edge_iter})])
            edge_iter += 1
            line = f.readline()
        f.close()
        self.G.add_weighted_edges_from([(self.sem[0], self.sem[1], {'r': 0, 'i': 0, 'num': edge_iter})])

    def get_resistance(self):
        R = [(r['num'],r['r']) for r in [w for u, v, w in self.G.edges.data('weight')]]
        return R

    def get_current(self):
        R = [r['r'] for r in [w for u, v, w in self.G.edges.data('weight')]]
        return R

    def solve(self):
        print(self.G.nodes)
        print(self.G.edges.data())

    def plot(self):
        print(self.G.edges.data())
        colors = self.get_current()
        pos = nx.circular_layout(self.G)
        nx.draw(self.G.to_undirected(), pos, edge_color=colors, width=4, edge_cmap=plt.jet(), with_labels = True)
        plt.plot()
        plt.show()


if __name__ == "__main__":
    sol = Solver()
    sol.load()
    sol.plot()