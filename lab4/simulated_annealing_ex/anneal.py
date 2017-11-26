import random
import math
import matplotlib.pyplot as plt
import numpy as np


class SimAnn():
    def __init__(self, N, C, T, stop_T, stop_I, alpha, fun, fun1):
        # graph arguments
        self.N = N
        self.C = C
        self.cities = fun(N, C)
        self.distances = create_distance_matrix(self.cities)
        # simann arguments
        self.T = T
        self.stopping_temperature = stop_T
        self.stopping_iter = stop_I
        self.alpha = alpha
        # solution args
        self.tour = self.nn_solution() if fun1 else self.random_solution() 
        self.best = self.tour
        self.best_fit = self.get_fitness(self.best)
        self.fit_list = []
        self.temp_list = []
        self.best_list = []
        self.init_tour = self.tour

    def random_solution(self):
        path = list(self.cities)
        random.shuffle(path)
        path.append(path[0])
        return path

    def nn_solution(self):
        path = []
        # process random starting node
        nodes_cpy = list(self.cities)
        tmp_node = random.choice(self.cities)
        path.append(tmp_node)
        nodes_cpy.remove(tmp_node)

        # process other nodes
        while nodes_cpy:
            nn_dist = min([self.distances[tmp_node[2]][neigh[2]] for neigh in nodes_cpy])
            nn_index = self.distances[tmp_node[2]].index(nn_dist)
            tmp_node = [n for n in nodes_cpy if n[2] == nn_index][0]
            nodes_cpy.remove(tmp_node)
            path.append(tmp_node)
        path.append(path[0])  # get back to starting point
        return path

    def get_fitness(self, path):  # total distance of path
        return sum([self.distances[path[i][2]][path[i + 1][2]] for i in range(len(path) - 1)])

    def sim_ann(self, swap_fun):
        iteration = 0
        while self.T >= self.stopping_temperature and iteration < self.stopping_iter:
            candidate = list(self.tour)
            swap_fun(candidate, len(self.tour))

            self.accept(candidate)
            self.T *= self.alpha 
            iteration += 1
            self.fit_list.append(self.get_fitness(self.tour))
            self.best_list.append(self.best_fit)
            self.temp_list.append(self.T)
        if self.best[len(self.best)-1] != self.best[0]:
            self.best.append(self.best[0])
            self.best_fit += distance(self.best[0], self.best[len(self.best)-1])
        print("Fitness : " + str(self.best_fit))
          

    def accept(self, candidate):
        path_fit = self.get_fitness(self.tour)
        candidate_fit = self.get_fitness(candidate)

        if candidate_fit < path_fit:
            if candidate_fit < self.best_fit:
                self.best_fit = candidate_fit
                self.best = candidate
            self.tour = candidate

        else:
            if random.random() < acc_prob(path_fit, candidate_fit, self.T):
                self.tour = candidate

    def plot_learning(self):
        x = [i for i in range(len(self.fit_list))]
        plt.plot(x, self.fit_list, label='fitness')
        plt.plot(x, self.best_list, label='shortest')

        plt.ylabel('Fitness')
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()

    def plot_temp(self):
        x = [i for i in range(len(self.temp_list))]
        plt.plot(x, self.temp_list, label='temperature')

        plt.ylabel('Temperature')
        plt.xlabel('Iteration')
        plt.legend()
        plt.show()

        
        
def swap_many(candidate, C):
    u = random.randint(1, C - 1)
    v = random.randint(1, C - 1)
    if u > v:
        u, v = v, u
        candidate[u:v] = reversed(candidate[u:v])
        
        
def swap_one(candidate, C):
    u = random.randint(0, C - 1)
    v = random.randint(0, C - 1)
    candidate[u], candidate[v] = candidate[v], candidate[u]
        
def acc_prob(path_fit, candidate_fit, t):
    return math.exp(-abs(candidate_fit - path_fit) / t)


def generate_random_cities(n, i):
    return [(round(random.uniform(-n, n), 0), round(random.uniform(-n, n), 0), j) for j in range(i)]


def generate_4_communities(n, c):
    cp = int(c / 4)
    sn = n / 2

    x1 = [(round(random.uniform(sn, n), 0), round(random.uniform(sn, n), 0), j) for j in range(cp)]
    x2 = [(round(random.uniform(-n, -sn), 0), round(random.uniform(sn, n), 0), j + cp) for j in range(cp)]
    x3 = [(round(random.uniform(-n, -sn), 0), round(random.uniform(-n, -sn), 0), j + 2 * cp) for j in range(cp)]
    x4 = [(round(random.uniform(sn, n), 0), round(random.uniform(-n, -sn), 0), j + 3 * cp) for j in range(cp)]
    return x1 + x2 + x3 + x4

def generate_9_communities(n, c):
    x = -n
    dx = int (n / 3)
    sn = int (n / 4.5)
    cp = int (c / 9)
    d = 0
    
    x1 = [(round(random.uniform(x, x + sn), 0), round(random.uniform(x, x + sn), 0), j) for j in range(cp)]
    d += len(x1)
    x += dx
    
    x2 = [(round(random.uniform(x, x + sn), 0), round(random.uniform(x, x + sn), 0), j + d) for j in range(cp)]
    d += len(x2)
    x += dx
    
    x3 = [(round(random.uniform(x, x + sn), 0), round(random.uniform(x, x + sn), 0), j + d) for j in range(cp)]
    d += len(x3)
    x += dx
    
    x4 = [(round(random.uniform(x, x + sn), 0), round(random.uniform(x, x + sn), 0), j + d) for j in range(cp)]
    d += len(x4)
    x += dx
    
    x5 = [(round(random.uniform(x, x + sn), 0), round(random.uniform(x, x + sn), 0), j + d) for j in range(cp)]
    d += len(x5)
    x += dx
    
    x6 = [(round(random.uniform(x, x + sn), 0), round(random.uniform(x, x + sn), 0), j + d) for j in range(cp)]
    d += len(x6)
    x += dx
    
    x7 = [(round(random.uniform(x, x + sn), 0), round(random.uniform(x, x + sn), 0), j + d) for j in range(cp)]
    d += len(x7)
    x += dx
    
    x8 = [(round(random.uniform(x, x + sn), 0), round(random.uniform(x, x + sn), 0), j + d) for j in range(cp)]
    d += len(x8)
    x += dx
    
    x9 = [(round(random.uniform(x, x + sn), 0), round(random.uniform(x, x + sn), 0), j + d) for j in range(cp)]
    d += len(x9)
    x += dx
    
    return x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9


def distance(coord1, coord2):
    return round(math.sqrt(math.pow(coord1[0] - coord2[0], 2) + math.pow(coord1[1] - coord2[1], 2)), 4)


def create_distance_matrix(nodes):
    n = len(nodes)
    result = [[distance(nodes[i], nodes[j]) for i in range(n)] for j in range(n)]
    return result


def plot_tour(path, N):
    N += math.sqrt(N)
    x = [c[0] for c in path]
    y = [c[1] for c in path]
    plt.plot(x, y, 'ro')
    plt.plot(x, y, 'b--')
    plt.axis([-N, N, -N, N])
    plt.show()
