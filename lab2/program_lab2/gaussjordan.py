import numpy as np
import numpy.linalg as np2
from timeit import default_timer as timer


def gepp(A, b):
    n = len(A)
    for pivot_row in range(n - 1):
        max_index = abs(A[pivot_row:, pivot_row]).argmax() + pivot_row  # Partial Pivoting
        if A[max_index, pivot_row] == 0:
            raise ValueError("Matrix is singular.")
        if max_index != pivot_row:  # swap
            A[[pivot_row, max_index]] = A[[max_index, pivot_row]]
            b[[pivot_row, max_index]] = b[[max_index, pivot_row]]
        for row in range(pivot_row + 1, n):  # Eliminate
            multiplier = A[row][pivot_row] / A[pivot_row][pivot_row]
            A[row][pivot_row] = multiplier
            for col in range(pivot_row + 1, n):
                A[row][col] = A[row][col] - multiplier * A[pivot_row][col]
            b[row] = b[row] - multiplier * b[pivot_row]

    x = np.zeros(n)
    k = n - 1
    while k >= 0:
        x[k] = (b[k] - np.dot(A[k, k + 1:], x[k + 1:])) / A[k, k]  # row k + columns from k+1 to n
        k = k - 1
    return x


def lup(A):
    n = len(A)
    for pivot_row in range(n - 1):
        max_index = abs(A[pivot_row:, pivot_row]).argmax() + pivot_row  # Partial Pivoting
        if A[max_index, pivot_row] == 0:
            raise ValueError("Matrix is singular.")
        if max_index != pivot_row:  # swap
            A[[pivot_row, max_index]] = A[[max_index, pivot_row]]
            b[[pivot_row, max_index]] = b[[max_index, pivot_row]]
        for row in range(pivot_row + 1, n):  # Eliminate
            multiplier = A[row][pivot_row] / A[pivot_row][pivot_row]
            A[row][pivot_row] = multiplier  # LU
            for col in range(pivot_row + 1, n):
                A[row][col] = A[row][col] - multiplier * A[pivot_row][col]
    return A


if __name__ == "__main__":
    a = np.array([[1., -1., 1., -1.], [1., 0., 0., 0.], [1., 1., 1., 1.], [1., 2., 4., 8.]])
    b = np.array([[14.], [4.], [2.], [2.]])
    a0 = a[2:,2]
    a1 = a[2:,2:]
    #maxes = a[0:,0].max(axis=1)
    #print(a[0:,0])
    #print((a[0:,0]/maxes).argmax())
    #[row/row.max() i=for row in A]
    # x2 = lup(a)
    # L = np.tril(a)
    # U = np.triu(a)
    # np.fill_diagonal(L,1)
    # print(L.dot(U))

