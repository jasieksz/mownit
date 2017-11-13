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

    def get_current(self):
        R = [r['i'] for r in [w for u, v, w in self.G.edges.data('weight')]]
        return R

    def sem_cycles(self):
        i = 0
        for cycle in nx.cycle_basis(self.G.to_undirected()):
            if (self.sem[0] in cycle and self.sem[1] in cycle):
                i += 1
        return i

    def solveKirchhoff(self):
        size = (self.sem_cycles() + self.G.number_of_nodes() + 1)
        current_row = 0
        A = np.zeros(shape=(size, self.G.number_of_edges()))
        B = np.zeros(size - 1)
        for node in self.G.nodes:
            for succ in self.G.successors(node):
                A[node][self.G.edges[(node, succ)]['weight']['num']] = -1
            for pred in self.G.predecessors(node):
                A[node][self.G.edges[(pred, node)]['weight']['num']] = 1
            current_row += 1
        if np.count_nonzero(A[0]) == 0:
            A = A[1:]
            current_row += 1
        for cycle in nx.cycle_basis(self.G.to_undirected()):
            if (self.sem[0] in cycle and self.sem[1] in cycle):
                for i in range(len(cycle) - 1):
                    u = cycle[i]
                    v = cycle[i + 1]
                    if self.G.has_edge(u, v):
                        A[current_row - 1][self.G.edges[(u, v)]['weight']['num']] = self.G.edges[(u, v)]['weight']['r']
                        B[current_row - 1] = self.sem[2]
                u = cycle[len(cycle) - 1]
                v = cycle[0]
                if self.G.has_edge(u, v):
                    A[current_row - 1][self.G.edges[(u, v)]['weight']['num']] = self.G.edges[(u, v)]['weight']['r']
                    B[current_row - 1] = self.sem[2]
            current_row += 1

        j = 0
        for i in range(0, len(A)):
            if (np.count_nonzero(A[i-j]) == 0 and A.shape[0] > B.shape[0]):
                A = np.delete(A, i-j, 0)
                j += 1

        I = np.linalg.lstsq(A, B)
        for q in [w for u, v, w in self.G.edges.data('weight')]:
            q['i'] = abs(I[0][q['num']])
        return I

    def plot(self):
        colors = self.get_current()
        pos = nx.circular_layout(self.G)
        nx.draw(self.G.to_undirected(), pos, edge_color=colors, width=4, edge_cmap=plt.cm.jet, with_labels=True)
        nx.draw_networkx_edge_labels(self.G, pos)
        plt.plot()
        plt.show()


def generateRandomGraph(number_of_nodes, maxDegreeOutFromVertex=2, maxResistance=50):
    G = {}
    for u in range(number_of_nodes):
        for j in range(random.randint(1, maxDegreeOutFromVertex)):
            v = u
            while (v == u or (u, v) in G or (v, u) in G):
                v = random.randint(0, number_of_nodes - 1)
            G[(u, v)] = random.random() * maxResistance
    return G


def writeToFile(G, filename):
    a = []
    for key in G.keys():
        a.append([*key, G[key]])
    a.sort()

    with open(filename, 'w') as f:
        f.write("""
    """.join(list(map(lambda x: ' '.join(list(map(lambda el: str(el), x))), a))))


if __name__ == "__main__":
    sol = Solver()
    sol.load()
    print(sol.solveKirchhoff())
    sol.plot()
