import numpy as np
import networkx as nx
import sys
import re


class Obwod:
    def __init__(self):
        self.graph = nx.DiGraph()  # utworzenie grafu

    def load(self, filename):
        f = open(filename, 'r')
        line = f.readline()  # odczytanie z linii SEM
        edge_num = 0  # numerowanie krawedzi
        self.sem = line.split(" ")  # SEM
        self.sem[0] = int(self.sem[0])  # Wierzcholek 1
        self.sem[1] = int(self.sem[1])  # Wierzcholek 2
        self.sem[2] = float(self.sem[2])  # Napiecie
        line = f.readline()
        while line and line not in ["", "\n"]:  # Kolejne krawedzie
            splitted = line.split(" ")
            u = int(splitted[0])
            v = int(splitted[1])
            w = float(splitted[2])  # Opor miedzy wierzcholkami
            self.graph.add_edge(u, v, weight=w, edge_num=edge_num)
            edge_num += 1
            line = f.readline()
        f.close()

    def solve(self, directed):
        a = []
        b = []
        edges_num = nx.get_edge_attributes(self.graph, 'edge_num')  # numery krawedzi
        total_edges = self.graph.number_of_edges()
        for node in self.graph:
            a.append([0] * total_edges)
            b.append([0])
            for neigh in self.graph.successors(node):  # I out
                edge = (node, neigh)
                a[-1][edges_num[edge]] += -1

            for non_neigh in self.graph.predecessors(node):  # I in
                a[-1][edges_num[(non_neigh, node)]] += 1

        weights = nx.get_edge_attributes(self.graph, 'weight')
        # wyszukanie cyklu SEM do ulozenia kolejnego rownania
        for cycle in nx.cycle_basis(self.graph.to_undirected()):
            cycle.append(cycle[0])
            if (self.sem[0] in cycle) and (self.sem[1] in cycle):
                b.append([self.sem[2]])  # suma(I*R) = U
                a.append([0] * total_edges)
                # cykl z SEM
                for i in range(0, len(cycle) - 1):
                    edge = (cycle[i], cycle[i + 1])
                    if edge not in edges_num:
                        edge = (cycle[i + 1], cycle[i])
                    a[-1][edges_num[edge]] = float(weights[edge])

        A = np.array(a)
        B = np.array(b)
        I = (np.linalg.lstsq(A, B))[0]  # rozwiazanie ukladu
        if (directed == "d"):  # krawedzie skierowane zgodnie z kierunkiem przeplywu pradu
            for edge in edges_num:
                if (I[edges_num[edge]] < 0):
                    I[edges_num[edge]] *= -1
                    w = float(weights[edge])
                    num = int(edges_num[edge])
                    self.graph.remove_edge(edge[0], edge[1])
                    self.graph.add_edge(edge[1], edge[0], weight=w, edge_num=num)

        edges_num = nx.get_edge_attributes(self.graph, 'edge_num')
        # dodanie atrybutow do wyswietlenia
        for edge in edges_num:
            self.graph[edge[0]][edge[1]]['I'] = I[edges_num[edge]]
            self.graph[edge[0]][edge[1]]['label'] = "R = " + str(
                self.graph[edge[0]][edge[1]]['weight']) + " I = " + str(round(I[edges_num[(edge[0], edge[1])]][0], 3))

        if (self.sem[0], self.sem[1]) in edges_num:
            self.graph[self.sem[0]][self.sem[1]]['label'] += " SEM = " + str(self.sem[2])
        else:
            self.graph[self.sem[1]][self.sem[0]]['label'] += " SEM = " + str(self.sem[2])

    def draw(self, filename):

        graphviz_graph = nx.to_agraph(self.graph)
        graphviz_graph.edge_attr['dir'] = 'forward'
        graphviz_graph.graph_attr['overlap'] = 'prism'
        graphviz_graph.layout(prog='dot')
        filename += ".png"
        graphviz_graph.draw(filename)


# if __name__ == "__main__":
#
#     rozw = Obwod()
#     if (len(sys.argv) != 3 and len(sys.argv) != 4):
#         print
#         'Uruchomienie: python Obwod.py <plik_wejsciowy> <plik_wyjsciowy> [d] - skierowanie zgodnie z przeplywem'
#     else:
#         rozw.load(sys.argv[1])
#         if (len(sys.argv) == 4):
#             rozw.solve(sys.argv[3])
#         else:
#             rozw.solve(0)
#         rozw.draw(sys.argv[2])



