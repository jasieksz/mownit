import numpy as np
import math
import matplotlib.pyplot as plt
import random


class Sudoku:
    def __init__(self, path="", t=10, stop_t=0.000001, stop_i=1000, alpha=0.95):
        self.t = t
        self.stop_t = stop_t
        self.stop_i = stop_i
        self.alpha = alpha

        self.initS = read_sudoku(path) if path != "" else generate_sudoku()
        self.initE = self.total_energy(self.initS)

        self.bestS = np.copy(self.initS)
        self.bestE = self.initE

        self.sud = np.copy(self.initS)
        self.energy = self.initE

        self.fitness_list = [self.initE]
        self.temp_list = [self.t]
        self.best_list = [self.bestE]
        self.iter_when_solution = 0

    def box_energy(self, sudoku, row, col):
        possible = [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3), (6, 6)]
        if not (row, col) in possible:
            raise Exception("Wrong index of box")
        fq = np.zeros(9)
        for r in range(row, row + 3):
            for c in range(col, col + 3):
                fq[int(sudoku[r][c]) - 1] += 1
        be = 0
        for e in fq:
            if e > 1:
                be += e
        return be

    def row_energy(self, sudoku, row):
        fq = np.zeros(9)
        for col in range(9):
            fq[int(sudoku[row][col]) - 1] += 1
        re = 0
        for e in fq:
            if e > 1:
                re += e
        return re

    def col_energy(self, sudoku, col):
        fq = np.zeros(9)
        for row in range(9):
            fq[int(sudoku[row][col]) - 1] += 1
        ce = 0
        for e in fq:
            if e > 1:
                ce += e
        return ce

    def point_energy(self, sudoku, row, col):
        re = self.row_energy(sudoku, row)
        ce = self.col_energy(sudoku, col)
        rb, cb = get_box(row, col)
        be = self.box_energy(sudoku, rb, cb)
        return re + ce + be

    def total_energy(self, sudoku):
        e = 0
        boxes = [(0, 0), (0, 3), (0, 6), (3, 0), (3, 3), (3, 6), (6, 0), (6, 3), (6, 6)]
        for b in boxes:
            e += self.box_energy(sudoku, b[0], b[1])
        for r in range(9):
            e += self.row_energy(sudoku, r)
        for c in range(9):
            e += self.col_energy(sudoku, c)
        return e

    def swap_one(self, candidate):
        row = np.random.randint(0, 9)
        p = np.random.randint(0, 9)
        q = np.random.randint(0, 9)

        candidate[row][p], candidate[row][q] = candidate[row][q], candidate[row][p]
        return row, p, q

    def acc_prob(self, sudokuE, candidateE, t):
        return math.exp(-abs(candidateE - sudokuE) / t)

    def anneal(self):
        iteration = 0
        while self.t >= self.stop_t and iteration < self.stop_i:

            candidateS = np.copy(self.sud)
            r, p, q = self.swap_one(candidateS)
            candidateE = self.energy \
                         - (self.point_energy(self.sud, r, p) + self.point_energy(self.sud, r, q)) \
                         + (self.point_energy(candidateS, r, p) + self.point_energy(candidateS, r, q))

            if candidateE < self.energy:
                if candidateE < self.bestE:
                    self.bestE = candidateE
                    self.bestS = np.copy(candidateS)
                self.sud = np.copy(candidateS)
                self.energy = candidateE

            else:
                if np.random.random() < self.acc_prob(self.energy, candidateE, self.t):
                    self.sud = np.copy(candidateS)
                    self.energy = candidateE

            self.t *= self.alpha
            iteration += 1

            if self.energy <= 6 and self.iter_when_solution == 0:
                self.iter_when_solution = iteration

            self.fitness_list.append(self.energy)
            self.best_list.append(self.bestE)
            self.temp_list.append(self.t)

    def plot_learning(self):
        x = [i for i in range(len(self.fitness_list))]
        plt.plot(x, self.fitness_list, label='fitness')
        plt.plot(x, self.best_list, label='best')

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


def read_sudoku(path):
    with open(path) as file:
        data = file.readlines()
        data = [[x for x in line.strip()] for line in data]
    n = len(data)
    for r in range(n):
        for c in range(n):
            if data[r][c] == 'x':
                data[r][c] = random.choice("123456789")  # get_not_in_row(data[r])
            data[r][c] = int(data[r][c])
    return np.array(data)


def get_not_in_row(row):
    a = [i for i in range(1, 10)]
    for e in row:
        if e != 'x':
            a.remove(int(e))
    return a[0]


def generate_sudoku():
    s = np.zeros((9, 9))
    for row in range(9):
        a = [i for i in range(1, 10)]
        for col in range(9):
            s[row][col] = np.random.choice(a)
            a.remove(s[row][col])
    return s


def generate_string_sudoku():
    s = generate_sudoku()
    r = ""
    for row in range(9):
        for col in range(9):
            r += str(int(s[row][col]))
        r += '\n'
    return r


def get_box(row, col):
    rb = row
    cb = col
    if row <= 2:
        rb = 0
    elif row <= 5:
        rb = 3
    elif row <= 8:
        rb = 6

    if col <= 2:
        cb = 0
    elif col <= 5:
        cb = 3
    elif col <= 8:
        cb = 6
    return rb, cb


if __name__ == '__main__':
    a = random.choice("123456789")
    print(a)
