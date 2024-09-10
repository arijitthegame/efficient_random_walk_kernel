import collections
import warnings

import numpy as np

from itertools import product

from numpy import ComplexWarning
from numpy.linalg import inv
from numpy.linalg import eig
from numpy.linalg import multi_dot
from scipy.linalg import expm
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator


def brute_force_random_walk_kernel(X, Y, lamda, p=None, kernel_type="exponential"):

    if p is not None:
        if type(p) is int and p > 0:
            if kernel_type == "geometric":
                mu_ = [1]
                fact = 1
                power = 1
                for k in range(1, p + 1):
                    fact *= k
                    power *= lamda
                    mu_.append(fact / power)
            else:
                mu_ = [1]
                power = 1
                for k in range(1, p + 1):
                    power *= lamda
                    mu_.append(power)
        else:
            raise TypeError(
                "p must be a positive integer bigger than " "zero or nonetype"
            )

    if lamda <= 0:
        raise TypeError("lambda must be positive bigger than equal")
    elif lamda > 0.5 and p is None:
        warnings.warn("random-walk series may fail to converge")

    XY = np.kron(X, Y)

    # algorithm presented in
    # [Kashima et al., 2003; Gartner et al., 2003]
    # complexity of O(|V|^6)

    # XY is a square matrix
    s = XY.shape[0]

    if p is not None:
        P = np.eye(XY.shape[0])
        S = mu_[0] * P
        for k in mu_[1:]:
            P = np.matmul(P, XY)
            S += k * P
    else:
        if kernel_type == "geometric":
            S = inv(np.identity(s) - lamda * XY).T
        elif kernel_type == "exponential":
            S = expm(lamda * XY).T

    return np.sum(S)


# TODO : AM I off by a factor of 1/p^2?


def conjugate_gradient_random_walk(X, Y, lamda, tol=1e-5, maxiter=100):

    xs, ys = X.shape[0], Y.shape[0]
    mn = xs * ys

    def lsf(x, lamda):
        xm = x.reshape((xs, ys), order="F")
        y = np.reshape(multi_dot((X, xm, Y)), (mn,), order="F")
        return x - lamda * y

        # A*x=b

    A = LinearOperator((mn, mn), matvec=lambda x: lsf(x, lamda))
    b = np.ones(mn)
    x_sol, _ = cg(
        A,
        b,
        rtol=tol,
        maxiter=maxiter,
    )
    return np.sum(x_sol)


def invert(w, v):
    return (np.real(np.sum(v, axis=0)), np.real(w))


def sd(x):
    return invert(*eig(x))


def spectral_decomposition_random_walk(X, Y, lamda, p=None, kernel_type="exponential"):

    assert (p is not None) or (kernel_type == "exponential")

    X = sd(X)
    Y = sd(Y)

    if p is not None:
        if type(p) is int and p > 0:
            if kernel_type == "geometric":
                mu_ = [1]
                fact = 1
                power = 1
                for k in range(1, p + 1):
                    fact *= k
                    power *= lamda
                    mu_.append(fact / power)
            else:
                mu_ = [1]
                power = 1
                for k in range(1, p + 1):
                    power *= lamda
                    mu_.append(power)
        else:
            raise TypeError(
                "p must be a positive integer bigger than " "zero or nonetype"
            )
    qi_Pi, wi = X
    qj_Pj, wj = Y

    # calculate flanking factor
    ff = np.expand_dims(np.kron(qi_Pi, qj_Pj), axis=0)

    # calculate D based on the method
    Dij = np.kron(wi, wj)
    if p is not None:
        D = np.ones(shape=(Dij.shape[0],))
        S = mu_[0] * D
        for k in mu_[1:]:
            D *= Dij
            S += k * D
            S = np.diagflat(S)
    else:
        # Exponential
        S = np.diagflat(np.exp(lamda * Dij))
    return ff.dot(S).dot(ff.T)


def sylvester_random_walk():
    pass


def fixed_point_random_walk():
    pass
