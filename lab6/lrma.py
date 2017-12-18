import numpy as np


def reduce(r, U, s, V):
    s = s[:r]  # r first singular values
    U = U[:, :r]
    V = V[:r, :]
    return U, s, V


def compose(U, s, V):
    D = np.diag(s)
    return U @ D @ V


def compress(data, k):
    U, s, V = np.linalg.svd(data)
    Ur, sr, Vr = reduce(k, U, s, V)
    result = compose(Ur, sr, Vr)
    return result