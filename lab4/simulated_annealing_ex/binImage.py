import numpy as np
import matplotlib.pyplot as plt
import math


def generate_image(n, d):
    return np.array(np.random.binomial(1, d, (n, n)))


def plot_image(image, size):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    ax.matshow(image, aspect="auto")
    plt.show()


def possible(row, col, n):
    if row >= n or row < 0:
        return False
    if col >= n or col < 0:
        return False
    return True


class BinImage:
    def __init__(self, n=20, density=0.3, neighbours=(0, 0), t=10, stop_t=0.000001, stop_i=10, alpha=0.95):
        self.n = n
        self.neighbours = neighbours

        self.initI = generate_image(self.n, density)
        self.initE = self.total_energy(self.initI)

        self.image = self.initI
        self.energy = self.initE

        self.bestI = self.initI
        self.bestE = self.initE

        self.t = t
        self.stop_t = stop_t
        self.stop_i = stop_i
        self.alpha = alpha

        self.fitness_list = [self.initE]
        self.temp_list = [self.t]
        self.best_list = [self.bestE]

    def point_energy(self, image, row, col):
        e = 0
        if image[row][col] == 1:
            for ngh in self.neighbours:
                rn = row + ngh[0]
                cn = col + ngh[1]
                if possible(rn, cn, len(image)) and image[rn][cn] == 1:
                    e += 1
        return e

    def total_energy(self, image):
        e = 0
        for row in range(len(image)):
            for col in range(len(image)):
                e += self.point_energy(image, row, col)
        return e

    def swap_one(self, candidate):
        s = np.random.randint(0, self.n)
        t = np.random.randint(0, self.n)
        p = np.random.randint(0, self.n)
        q = np.random.randint(0, self.n)
        candidate[s][t], candidate[p][q] = candidate[p][q], candidate[s][t]
        return s, t, p, q


    def acc_prob(self, imageE, candidateE, t):
        return math.exp(-abs(candidateE - imageE) / t)

    def anneal(self):
        iteration = 0
        while self.t >= self.stop_t and iteration < self.stop_i:
            candidateI = np.copy(self.image)
            s, t, p, q = self.swap_one(candidateI)
            candidateE = self.energy \
                         - (self.point_energy(self.image, s, t) + self.point_energy(self.image, p, q)) \
                         + (self.point_energy(candidateI, s, t) + self.point_energy(candidateI, p, q))


            if candidateE < self.energy:
                if candidateE < self.bestE:
                    self.bestE = candidateE
                    self.bestI = np.copy(candidateI)
                self.image = np.copy(candidateI)
                self.energy = candidateE

            else:
                if np.random.random() < self.acc_prob(self.energy, candidateE, self.t):
                    self.image = np.copy(candidateI)
                    self.energy = candidateE

            self.t *= self.alpha
            iteration += 1

            self.fitness_list.append(self.energy)
            self.best_list.append(self.bestE)
            self.temp_list.append(self.t)

    def plot_learning(self):
        x = [i for i in range(len(self.fitness_list))]
        plt.plot(x, self.fitness_list, label='fitness')
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


if __name__ == '__main__':
    n1 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    n2 = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, -1), (-1, 1)]

    bim = BinImage(n=64, density=0.3, neighbours=n1, t=128, stop_t=0.0000001, stop_i=100000, alpha=0.9995)
    bim.anneal()

    plot_image(bim.initI, 5)
    print(bim.initE)

    plot_image(bim.bestI, 5)
    print(bim.bestE)

    bim.plot_learning()
    bim.plot_temp()
