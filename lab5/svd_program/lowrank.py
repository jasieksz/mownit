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


def compress(image, k):
    U, s, V = np.linalg.svd(image)
    Ur, sr, Vr = reduce(k, U, s, V)
    result = compose(Ur, sr, Vr)
    return result


if __name__ == "__main__":
    face = scipy.misc.face(gray=True)
    compressed = [compress(face, k) for k in [8, 32, 64, 128, 256, 512]]
    diff = [np.abs(face - compressed[i]) for i in range(len(compressed))]
    plt.imshow(compressed[1])
    plt.show()
