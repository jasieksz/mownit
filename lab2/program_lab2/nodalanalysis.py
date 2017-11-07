import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np

# nx.draw(G,edge_color=aw, width = 5, edge_cmap=plt.cm.jet, with_labels=True) #edge_color = wagi kolejnych krawedzi

G = nx.DiGraph()
for i in range(1, 4):
    G.add_node(i)
edges = [(1, 2, {'r': 1., 'i': 0}), (2, 3, {'r': 2., 'i': 0}), (3, 1,{'r': 0., 'i': 0})]
G.add_weighted_edges_from(edges)

D = [w for u, v, w in G.edges.data('weight')]
R = [r['r'] for r in D]
for i in W:
    print(i['r'])
print(W)
# nx.draw(G, with_labels=True)
# plt.plot()
# plt.show()
