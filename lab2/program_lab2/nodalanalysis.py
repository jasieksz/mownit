import networkx as nx
import matplotlib.pyplot as plt
import math
import numpy as np
import csv
import circuitgenerator


class CircutSolver:
    def __init__(self):
        self.G = self.make_graph()
        self.emf = self.get_emf()
        self.G.add_edge(self.emf[0], self.emf[1], weight=0.)

    def make_graph(self):
        graph = nx.DiGraph()
        with open('data/circuit.csv', newline='') as circuit_csv:
            reader = csv.DictReader(circuit_csv)
            for row in reader:
                graph.add_edge(int(row['u']), int(row['v']), weight=float(row['r']))
        return graph

    def get_emf(self):
        with open('data/emf.csv', newline='') as emf_csv:
            reader = csv.DictReader(emf_csv)
            for row in reader:
                emf = (int(row['s']), int(row['t']), float(row['e']))
                return emf

    def get_edge_index(self, n1, n2):
        i = 0
        for u, v, w in self.G.edges.data():
            if (u == n1 and v == n2) or (u == n2 and v == n1):
                return i
            i += 1
        return -1

    def get_resistance(self):
        return [w['weight'] for u, v, w in self.G.edges.data()]

    def get_currents(self):
        return [w['I'] for u, v, w in self.G.edges.data()]

    def add_currents(self, I):
        i = 0
        for u, v, w in self.G.edges.data():
            w['I'] = I[i]
            i += 1

    def get_cycles(self):
        cycles = nx.cycle_basis(self.G.to_undirected())
        return [cycle for cycle in cycles if (self.emf[0] in cycle and self.emf[1] in cycle)]

    def make_equation(self):
        A = []
        B = []
        # FIRST LAW
        for node in self.G.nodes:
            A.append([0.] * self.G.number_of_edges())
            B.append(0.)
            for succ in self.G.successors(node):
                A[-1][self.get_edge_index(node, succ)] -= 1
            for pred in self.G.predecessors(node):
                A[-1][self.get_edge_index(pred, node)] += 1
        # SECOND LAW
        for cycle in self.get_cycles():
            A.append([0.] * self.G.number_of_edges())
            B.append(0.)
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]  # cycle is [ 0 1 2 4 ] , % is for edge 4 -> 0
                num = self.get_edge_index(u, v)
                r = self.get_resistance()[num]
                A[-1][num] = r
            B[-1] += self.emf[2]
        return A, B

    def solve(self):
        A, B = self.make_equation()
        A = np.array(A)
        B = np.array(B)
        print(B.shape)
        I = np.linalg.lstsq(A, B)
        # I = np.linalg.solve(np.dot(A.transpose(), A), np.dot(A.transpose(), B))
        result = [abs(i) for i in I[0]]
        self.add_currents(result)
        return result

    def add_emf_label(self):
        for u, v, w in self.G.edges.data():
            if (u == self.emf[0] and v == self.emf[1]) or (v == self.emf[0] and u == self.emf[1]):
                w['EMF'] = self.emf[2]

    def round_i_label(self):
        for u, v, w in self.G.edges.data():
            w['I'] = (math.ceil(w['I'] * 1000) / 1000)

    def plot_graph(self):
        self.add_emf_label()
        colors = self.get_currents()
        self.round_i_label()

        pos = nx.circular_layout(self.G)  # nx.fruchterman_reingold_layout(self.G)
        plt.figure(figsize=(10, 10))
        nx.draw(self.G.to_undirected(), pos, width=2, with_labels=True, edge_color=colors, edge_cmap=plt.cm.jet)
        nx.draw_networkx_edge_labels(self.G, pos)
        plt.show()


if __name__ == "__main__":
    gen = circuitgenerator.CircuitGenerator(0, 4, 12, 6, 'connected')
    gen.generate_emf()
    gen.generate_circuit()

    sol = CircutSolver()
    sol.solve()
    sol.plot_graph()
