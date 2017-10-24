import networkx as nx
import matplotlib.pyplot as plt
import random

a = [(random.randint(0,100), random.randint(0,100), random.randint(0,100)) for _ in range(20)]
aw = [c for _,_,c in a]

G = nx.Graph()
G.add_weighted_edges_from(a)

print(len(G.edges),len(aw))

nx.draw(G,edge_color=aw, width = 5, edge_cmap=plt.cm.jet, with_labels=True) #edge_color = wagi kolejnych krawedzi
plt.show()

# colors = range(20)
# nx.draw(G, node_color='#A0CBE2', edge_color=colors,
#         width=4, edge_cmap=plt.cm.Greens, with_labels=False)
# plt.show()
