import numpy as np
import random as rnd
import time
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

from __future__ import division
from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot
from math import sqrt
import pprint
import scipy
import scipy.linalg

# Define some global constants
SIGMA = 0.1
LAMBDA_COEFF = 0.1
P_HALT = 0.01
NB_RANDOM_WALKS = 2000
BIG_NUMBER = 1000  # 10000
RANDOM_SEED = 180
BLOCK_SIZE = 20


np.random.seed(RANDOM_SEED)
t_variables = np.random.uniform(size=(2 * BIG_NUMBER, 2 * BIG_NUMBER))
g_variables = np.where(
    np.random.normal(size=(2 * BIG_NUMBER, 2 * BIG_NUMBER)) > 0.0, 1.0, -1.0
)


def adj_matrix_to_lists(A):
    adj_lists = []
    weight_lists = []
    for i in range(len(A)):
        neighbors = []
        weights = []
        visits = []
        for j in range(len(A[i])):
            if A[i][j] > 0.0:
                neighbors.append(j)
                weights.append(A[i][j])
        adj_lists.append(neighbors)
        weight_lists.append(weights)
    return adj_lists, weight_lists


"Some modulation functions to use"


def f_func_d_reg_dis1(i):
    "modulation func for d-regularised Lap w d=1"
    i = int(i)
    if i == 0:
        return 1.0
    else:
        return scipy.special.factorial2(2 * i - 1) / (scipy.special.factorial2(2 * i))


def f_func_d_reg_dis2(i):
    "Modulation func for d-regularised Lap with d=2"
    i = int(i)
    return 1.0


def f_func_diffusion(i, lambda_coeff):
    "modulation function for diffusion kernel"
    return lambda_coeff ** i / (2 ** i * scipy.special.factorial(i))


def f_func_p_step_rw_pis2(i):
    "modulation function for p-step RW kernel"
    return scipy.special.binom(1, i)


def exact_graph_kernel_value(G1, G2, v1, v2, w1, w2l, lambda_coeff=LAMBDA_COEFF):
    G1_flat = np.reshape(G1, (-1))
    G2_flat = np.reshape(G2, (-1))
    G1_G2_cartesian_product = np.einsum("N,M->NM", G1_flat, G2_flat)
    G1_G2_cartesian_product_unflattened = np.reshape(
        G1_G2_cartesian_product, (len(G1), len(G1), len(G2), len(G2))
    )
    G1_G2_final = np.transpose(G1_G2_cartesian_product_unflattened, (0, 2, 1, 3))
    G1_G2_final = np.reshape(G1_G2_final, (len(G1) * len(G2), len(G1) * len(G2)))
    G1_G2_final = scipy.linalg.expm(lambda_coeff * G1_G2_final)
    v = np.reshape(np.einsum("x,y->xy", v1, v2), (-1))
    w = np.reshape(np.einsum("x,y->xy", w1, w2), (-1))
    return np.einsum("a,ab,b->", v, G1_G2_final, w)


def create_pq_vectors(
    adj_lists,
    weight_lists,
    anchor_points_dict,
    p_halt,
    nb_random_walks,
    f,
    is_left,
    base_nb_walk_index,
):

    s_matrix = np.zeros((nb_random_walks, len(anchor_points_dict), len(adj_lists)))
    for w in range(nb_random_walks):
        for k in range(len(adj_lists)):
            load = 1.0
            step_counter = 0
            current_vertex = k
            x_index = is_left * BIG_NUMBER + step_counter
            y_index = is_left * BIG_NUMBER + w + base_nb_walk_index
            if current_vertex in anchor_points_dict.keys():
                add_term = load * np.sqrt(f(step_counter))
                add_term *= g_variables[x_index][y_index]
                s_matrix[w][anchor_points_dict[current_vertex]][k] += add_term
            if adj_lists[current_vertex] == []:
                break
            while t_variables[x_index][y_index] > p_halt:
                rnd_index = int(rnd.uniform(0, 1) * len(adj_lists[current_vertex]))
                multiplier = len(adj_lists[current_vertex])
                load *= weight_lists[current_vertex][rnd_index]
                load *= multiplier / np.sqrt(1.0 - p_halt)
                step_counter += 1
                current_vertex = adj_lists[current_vertex][rnd_index]
                if current_vertex in anchor_points_dict.keys():
                    x_index = is_left * BIG_NUMBER + step_counter
                    y_index = is_left * BIG_NUMBER + w + base_nb_walk_index
                    add_term = load * np.sqrt(f(step_counter))
                    add_term *= g_variables[x_index][y_index]
                    s_matrix[w][anchor_points_dict[current_vertex]][k] += add_term
    return s_matrix


def approximate_graph_kernel_value(
    G1,
    G2,
    v1,
    v2,
    w1,
    w2,
    anchor_fraction=1.0,
    base_nb_walk_index=0,
    kernel_type="exponential",
    lambda_coeff=LAMBDA_COEFF,
    p_halt=P_HALT,
    nb_random_walks=NB_RANDOM_WALKS,
):

    G1_adj_lists, G1_weight_lists = adj_matrix_to_lists(
        G1
    )  # TODO do we need list or matrix? instead of N^2 we can do this N times
    G2_adj_lists, G2_weight_lists = adj_matrix_to_lists(G2)
    G1_nb_anchor_points = int(anchor_fraction * len(G1_adj_lists))
    G1_anchor_points = np.random.choice(
        np.arange(len(G1_adj_lists)), size=G1_nb_anchor_points, replace=False
    )
    G1_anchor_points = np.sort(G1_anchor_points)
    G1_anchor_points_dict = dict(zip(G1_anchor_points, np.arange(G1_nb_anchor_points)))
    G2_nb_anchor_points = int(anchor_fraction * len(G2_adj_lists))
    G2_anchor_points = np.random.choice(
        np.arange(len(G2_adj_lists)), size=G2_nb_anchor_points, replace=False
    )
    G2_anchor_points = np.sort(G2_anchor_points)
    G2_anchor_points_dict = dict(zip(G2_anchor_points, np.arange(G2_nb_anchor_points)))

    f_function = None
    if kernel_type == "exponential":
        f_function = lambda i: f_func_diffusion(i, lambda_coeff)
    p1 = create_pq_vectors(
        G1_adj_lists,
        G1_weight_lists,
        G1_anchor_points_dict,
        p_halt=p_halt,
        nb_random_walks=nb_random_walks,
        f=f_function,
        is_left=0,
        base_nb_walk_index=base_nb_walk_index,
    )

    p2 = create_pq_vectors(
        G2_adj_lists,
        G2_weight_lists,
        G2_anchor_points_dict,
        p_halt=p_halt,
        nb_random_walks=nb_random_walks,
        f=f_function,
        is_left=0,
        base_nb_walk_index=base_nb_walk_index,
    )
    q1 = create_pq_vectors(
        G1_adj_lists,
        G1_weight_lists,
        G1_anchor_points_dict,
        p_halt=p_halt,
        nb_random_walks=nb_random_walks,
        f=f_function,
        is_left=1,
        base_nb_walk_index=base_nb_walk_index,
    )
    q2 = create_pq_vectors(
        G2_adj_lists,
        G2_weight_lists,
        G2_anchor_points_dict,
        p_halt=p_halt,
        nb_random_walks=nb_random_walks,
        f=f_function,
        is_left=1,
        base_nb_walk_index=base_nb_walk_index,
    )

    G1_latent_embedding = np.einsum(
        "br,br->br", np.einsum("brN,N->br", p1, v1), np.einsum("brN,N->br", q1, w1)
    )
    G2_latent_embedding = np.einsum(
        "br,br->br", np.einsum("brN,N->br", p2, v2), np.einsum("brN,N->br", q2, w2)
    )
    final_batch = np.einsum("bx,by->bxy", G1_latent_embedding, G2_latent_embedding)
    return (1.0 / nb_random_walks) * np.sum(final_batch)


# block variant
def approximate_graph_kernel_value_with_blocks(
    G1,
    G2,
    v1,
    v2,
    w1,
    w2,
    anchor_fraction=1.0,
    kernel_type="exponential",
    lambda_coeff=LAMBDA_COEFF,
    p_halt=P_HALT,
    nb_random_walks=NB_RANDOM_WALKS,
    block_size=NB_RANDOM_WALKS,
):
    approximate_value = 0
    for i in range(int(nb_random_walks / block_size)):
        approximate_value += approximate_graph_kernel_value(
            G1,
            G2,
            v1,
            v2,
            w1,
            w2,
            anchor_fraction=1.0,
            base_nb_walk_index=i * block_size,
            kernel_type="exponential",
            lambda_coeff=lambda_coeff,
            p_halt=P_HALT,
            nb_random_walks=block_size,
        )
    return approximate_value * block_size / nb_random_walks
