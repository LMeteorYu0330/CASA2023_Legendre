import taichi as ti
import numpy as np

n = 20
x_res = 100
dx = 2 / x_res
legendre_table = np.zeros((n, x_res))
alpha = np.zeros((n, n, n))


def calculateLegendrePolynomial():
    for nth in range(n):
        for x in range(x_res):
            x_ = -1 + x * dx
            if nth == 0:
                legendre_table[nth, x] = 1
            elif nth == 1:
                legendre_table[nth, x] = x_
            else:
                legendre_table[nth, x] = ((2 * nth - 1) * x_ * legendre_table[nth - 1, x] - (nth - 1) * legendre_table[
                    nth - 2, x]) / nth


def compress(vf: ti.template(), df: ti.template()):
    pass


calculateLegendrePolynomial()
print(legendre_table[3])
