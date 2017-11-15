import scipy.misc
import numpy as np
import matplotlib.pyplot as plt


def reduce(r, U, s, V):
    s = s[:r]  # r first singular values
    U = U[:, :r]
    V = V[:r, :]
    return U, s, V


def compose(U, s, V):
    D = np.diag(s)
    return U @ D @ V


if __name__ == "__main__":
    k = 64
    face = scipy.misc.face(gray=True)

    U, s, V = np.linalg.svd(face)
    Ur, sr, Vr = reduce(k, U, s, V)
    face2 = compose(Ur, sr, Vr)

    plt.imshow(face2)
    plt.show()
