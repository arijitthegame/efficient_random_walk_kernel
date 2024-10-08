import collections
import warnings

import numpy as np

from itertools import product

from numpy import ComplexWarning
from numpy.linalg import inv
from numpy.linalg import eig
from numpy.linalg import multi_dot
from scipy.linalg import expm, eigh
from scipy.sparse.linalg import cg
from scipy.sparse.linalg import LinearOperator

from scipy.optimize import fixed_point
from control import dlyap  # TODO : Remove dylap


def brute_force_random_walk_kernel(X, Y, lamda, p=None, kernel_type="exponential"):

    if p is not None:
        if type(p) is int and p > 0:
            if kernel_type == "exponential":
                mu_ = [1]
                fact = 1
                power = 1
                for k in range(1, p + 1):
                    fact *= k
                    power *= lamda
                    mu_.append(power / fact)
            elif kernel_type == "geometric":
                mu_ = [1]
                power = 1
                for k in range(1, p + 1):
                    power *= lamda
                    mu_.append(power)
            else:
                raise ValueError("Other powers are not implemented")
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
            if kernel_type == "exponential":
                mu_ = [1]
                fact = 1
                power = 1
                for k in range(1, p + 1):
                    fact *= k
                    power *= lamda
                    mu_.append(power / fact)
            elif kernel_type == "geometric":
                mu_ = [1]
                power = 1
                for k in range(1, p + 1):
                    power *= lamda
                    mu_.append(power)
            else:
                raise ValueError("Other power series are not yet implemented.")
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


# for some reason this is causing my colab to crash.
# TODO : figure out why that is the case
# If I can not figure it out, rewrite the equation so that it can be solved by scipy solver for Sylvester equation.


def sylvester_random_walk(X, Y, lamda):
    """
    Sylvester equation Methods (lyapunov)
    O(n^3)
    for graph with no labels only : else implement "COMPUTATION OF THE CANONICAL DECOMPOSITION BYMEANS OF A SIMULTANEOUS GENERALIZED SCHURDECOMPOSITION∗"
    with https://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.linalg.qz.html
    """
    # https://python-control.readthedocs.io/en/0.8.0/generated/control.dlyap.html

    n1 = X.shape[0]
    n2 = Y.shape[0]
    # in case the matrix is empty i add a diagonal, works great
    A1 = (X + X.T) / 2 + np.eye(n1) * 1e1
    A2 = (Y + Y.T) / 2 + np.eye(n2) * 1e1
    n = n1 * n2
    px = np.ones((n, 1)) / n  # np.ones((n,1))
    qx = np.ones((n, 1)) / n  # np.ones((n,1))
    M = dlyap(A=lamda * X, Q=Y, C=px.reshape((n1, n2)))
    return -1 * (qx.T @ M.reshape((-1, 1)))


def fixed_point_random_walk(X, Y, lamda, maxiter=1500):

    """This is technically slow as we are materializing the entire tensor product"""
    Wx = np.kron(X, Y)
    n = Wx.shape[0]
    px = np.ones((n, 1)) / n
    qx = np.ones((n, 1)) / n
    # diagonaliser
    Wx = (Wx + Wx.T) / 2

    if lamda >= 1 / abs(eigh(Wx, eigvals_only=True, eigvals=(n - 1, n - 1))[0]):
        print("Cannot converge. Choose a smaller value of lamda")
        raise ValueError()

    def func(x, px, lamda, Wx):
        return px + (lamda * Wx) @ x

    x = fixed_point(func, px, args=(px, lamda, Wx), maxiter=maxiter)

    k = np.real(qx.T @ x)
    return k


def fixed_point_kernel_fast(X, Y, lamda, maxiter=1500):
    """Faster variant of the above. Does not explicitly materialize the kronecker product
    For this algorithm to work properly, A1, A2 needs to be symmetric
    """
    xs, ys = X.shape[0], Y.shape[0]
    mn = xs * ys
    px = np.ones((mn)) / mn
    qx = np.ones((mn, 1)) / mn

    # check for convergence before running the algorithm
    eig1 = eigh(X, eigvals_only=True, eigvals=(xs - 1, xs - 1))[0]
    eig2 = eigh(Y, eigvals_only=True, eigvals=(ys - 1, ys - 1))[0]
    if lamda >= 1 / abs(eig1 * eig2):
        print("Cannot converge. Choose a smaller value of lamda")
        raise ValueError()

    def lsf(x, lamda):
        xm = x.reshape((xs, ys), order="F")
        y = np.reshape(multi_dot((X, xm, Y)), (mn,), order="F")
        return lamda * y

    A = LinearOperator((mn, mn), matvec=lambda x: lsf(x, lamda))

    def func(x, px, A):
        return px + A.matvec(x)

    x = fixed_point(func, np.zeros(mn), args=(px, A), maxiter=maxiter)

    k = np.real(qx.T @ x)
    return k
