import math
import random
import coordinates as co


class SimulatedAnneal(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        # kryterium stopu
        self.T = math.sqrt(self.coords.N) if T == -1 else T
        self.alpha = 0.995 if alpha == -1 else T
        self.stopping_temperature = 0.0001 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100 if stopping_iter == -1 else stopping_iter
        # iteracje
        self.iteration = 1
        self.tmp_result = self.init_solution()
        #self.best_result




if __name__ == "__main__":
    myCoords = co.Coordinates(20, 5)
    print(myCoords.coords)




    #
    #
    # z2) obraz 256x256 -> symulacja krystalizacji
    #       gestosc -> czarne / n^2
    #       rozne sasiedztwa , nie symetrzyczne
    #       rozne funkcje energi -> lubie czarne (suma) / lubie z prawej z lewej nie itd.
    #       brzegi przechodza na druga strone (cykl)
    #       WIZUALIZACJA !!! / wykresy energii / parametry wyżażania (testowac rozne wyzazania -> szybkie , liniowe, wolne)
    #
    # z3) SUDOKU
    #       cel -> czy umiemy dobrze dopasowac parametry poczatkowe (test na benchmarkach)
    #
    #
    #
    #
