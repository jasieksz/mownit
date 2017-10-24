import numpy as np
import numpy.linalg as np2
from timeit import default_timer as timer


def GENP(A, b):
    n = len(A)
    for pivot_row in range(n - 1):

        for row in range(pivot_row + 1, n):
            multiplier = A[row][pivot_row] / A[pivot_row][pivot_row]
            A[row][pivot_row] = multiplier
            for col in range(pivot_row + 1, n):
                A[row][col] = A[row][col] - multiplier * A[pivot_row][col]
            b[row] = b[row] - multiplier * b[pivot_row]

    x = np.zeros(n)
    k = n - 1
    x[k] = b[k] / A[k, k]

    while k >= 0:
        x[k] = (b[k] - np.dot(A[k, k + 1:], x[k + 1:])) / A[k, k]  # row k + columns from k+1 to n
        k = k - 1
    return x


if __name__ == "__main__":
    A = np.array([[1., -1., 1., -1.], [1., 0., 0., 0.], [1., 1., 1., 1.], [1., 2., 4., 8.]])
    b = np.array([[14.], [4.], [2.], [2.]])
    start = timer()
    GENP(np.copy(A),np.copy(b))
    end = timer()
    print(end - start)

    start = timer()
    np2.solve(A,b)
    end = timer()
    print(end - start)
