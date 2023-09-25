###
# Implementation of the model described in: https://arxiv.org/abs/nucl-th/9401015

def nerlo_pomorska_pomorski(Z, N):
    A = Z + N
    return 0.966 * (A**(1/3))* (1 - (0.182*(N-Z)/A) + 1.652/A)