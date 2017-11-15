from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


def my_plot(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z)
    plt.show()


def v_gen(s, t):  # s 0:2pi t 0:pi
    V = np.array([[np.cos(s) * np.sin(t)],
                  [np.sin(s) * np.sin(t)],
                  [np.cos(t)]])

    return V


def a_gen(sx, sy):
    A = np.random.randn(sx, sy)
    return A


def gen_angles(iter):
    s = [(2 * np.pi) * np.random.random_sample() for i in range(iter)]
    t = [np.pi * np.random.random_sample() for i in range(iter)]
    return s, t


def svd(A):
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return U, s, V


if __name__ == "__main__":
    points = 500
    s, t = gen_angles(points)
    X = [np.cos(s[i]) * np.sin(t[i]) for i in range(len(s))]
    Y = [np.sin(s[i]) * np.sin(t[i]) for i in range(len(s))]
    Z = [np.cos(t[i]) for i in range(len(s))]

    A1 = a_gen(3, 3)
    A2 = a_gen(3, 3)
    A3 = a_gen(3, 3)
    P = np.row_stack([X, Y, Z])
    B = A3 @ (A2 @ (A1 @ P))

    U1, s1, V1 = svd(A1)
    U2, s2, V2 = svd(A2)
    U3, s3, V3 = svd(A3)

    fig = plt.figure(figsize=(15, 15))
    ax = fig.gca(projection='3d')
    ax.scatter(X, Y, Z, color='orange')
    ax.scatter(B[0], B[1], B[2], color='g')

    x0, y0, z0 = [[0] * 3] * 3
    aX = [U1[0][0], U2[0][0], U3[0][0]]
    aY = [U1[0][1], U2[0][1], U3[0][1]]
    aZ = [U1[0][2], U2[0][2], U3[0][2]]
    ax.quiver(x0, y0, z0, aX, aY, aZ, color='r')

    plt.show()
