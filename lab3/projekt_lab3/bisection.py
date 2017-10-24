import math
import numpy as np
import random
import scipy.misc as sp


def binsearch(a, b, fun, epsilon, width, iterations):
    i = 0
    while (abs(b - a) > width):
        x = a + (b - a) / 2
        # print(x, fun(x))
        if (abs(fun(x)) < epsilon):
            print("ITERATIONS", i)
            return x
        else:
            if (np.sign(fun(x)) * np.sign(fun(b)) < 0):
                a = x
            else:
                b = x
        if (i == iterations):
            return math.inf
        i = i + 1
    return math.inf


def fun1(x):
    return math.cos(x) * math.cosh(x) - 1


def fun2(x):
    return (1 / x - math.tan(x))


def fun2p(x):
    return (-1 / math.pow(x, 2) - 1 / math.pow(math.cos(x), 2))


def fun3(x):
    return (math.pow(2, -x) + math.pow(math.e, x) + 2 * math.cos(x) - 6)


# print("BIN SEARCH TEST")
# print("X =",binsearch(math.pi * 1.5, math.pi * 2, fun1, 1e-10, 1e-10, 1000))
# print("X =", binsearch(0, math.pi * 0.5, fun2, 1e-10, 1e-10, 1000))
# print("X =",binsearch(1, 3, fun3, 1e-7, 1e-10, 1000))

def newraphs(x0, fun, funp, epsilon, iterations):
    i = 0
    while (i < iterations):
        y = fun(x0)
        yp = funp(x0)
        print(y,yp)
        if (abs(yp) < epsilon):
            return math.inf
        x1 = x0 - y / yp
        if (abs(x1 - x0) < epsilon):
            return x1
        x0 = x1
        i = i + 1
    return math.inf


print("NEWTON TEST")


def fun1p(x):
    return math.cos(x)*math.sinh(x) - math.sin(x)*math.cosh(x)

def fun3p(x):
    return math.pow(math.e,x)-math.pow(2,-x)*math.log(2,math.e)-2*math.sin(x)

print("X =",newraphs(0, fun1, fun1p, 1e-6, 100))
print("X =", newraphs(random.random()%2*math.pi, fun2, fun2p, 1e-6, 100))
print("X =",newraphs(2, fun3, fun3p, 1e-6, 100))
