import math
import random
import coordinates


class SimulatedAnneal(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        self.N = len(coords)
        self.distance_matrix = self.create_distance_matrix(coords)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.alpha = 0.995 if alpha == -1 else T
        self.stopping_temperature = 0.0001 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

    def distance(self, coord1, coord2):
        return round(math.sqrt(math.pow(coord1[0] - coord2[0], 2) + math.pow(coord1[1] - coord2[1], 2)), 4)

    def create_distance_matrix(self, coords):
        n = len(coords)
        result = [[self.distance(coords[i], coords[j]) for i in range(n)] for j in range(n)]
        return result


if __name__ == "__main__":
    coords = [[round(random.uniform(-10, 10), 4), round(random.uniform(-10, 10), 4)] for i in range(5)]
    print(coords)
