from __future__ import division
from pprint import pprint
import torch
import numpy as np
import random as rnd
import time


from numpy import array, zeros, diag, diagflat, dot
from math import sqrt
import pprint
import scipy
import scipy.linalg
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_dense_adj

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


from gvoys import (
    adj_matrix_to_lists,
    create_pq_vectors,
    f_func_diffusion,
)

SIGMA = 0.1
LAMBDA_COEFF = 0.1
P_HALT = 0.2
NB_RANDOM_WALKS = 100


def generate_approximate_graph_kernel_features(
    G1,
    v1,
    w1,
    anchor_fraction=1.0,
    base_nb_walk_index=0,
    kernel_type="exponential",
    lambda_coeff=LAMBDA_COEFF,
    p_halt=P_HALT,
    nb_random_walks=NB_RANDOM_WALKS,
) -> dict:
    """Using the similar construction as in GVOY to extract the graph features"""

    G1_adj_lists, G1_weight_lists = adj_matrix_to_lists(
        G1
    )  # TODO do we need list or matrix? instead of N^2 we can do this N times
    G1_nb_anchor_points = int(anchor_fraction * len(G1_adj_lists))
    G1_anchor_points = np.random.choice(
        np.arange(len(G1_adj_lists)), size=G1_nb_anchor_points, replace=False
    )
    G1_anchor_points = np.sort(G1_anchor_points)
    G1_anchor_points_dict = dict(zip(G1_anchor_points, np.arange(G1_nb_anchor_points)))

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

    G1_latent_embedding = np.einsum(
        "br,br->br", np.einsum("brN,N->br", p1, v1), np.einsum("brN,N->br", q1, w1)
    )

    graph_features = {}
    graph_features["latent_embedding"] = G1_latent_embedding
    graph_features["v"] = v1
    graph_features["w"] = w1
    return graph_features


def approximate_graph_kernel_value(
    G1_latent_embedding, G2_latent_embedding, nb_random_walks=NB_RANDOM_WALKS
) -> float:
    final_batch = np.einsum("bx,by->bxy", G1_latent_embedding, G2_latent_embedding)
    return (1.0 / nb_random_walks) * np.sum(final_batch)


"""Load dataset from TUDataset"""
dataset = TUDataset(root="data/TUDataset", name="MUTAG")

"""Processing the dataset"""
all_dataset = [to_dense_adj(data.edge_index).squeeze().numpy() for data in dataset]
full_labels = [data.y.data.item() for data in dataset]

"""collect data stats and use it to choose lambda so that the power series expresssion converges."""
max_degree = 1
max_degree_id = 0
max_edges_id = 0
max_nodes_id = 0
num_nodes = []
num_edges = []
for i in range(len(all_dataset)):
    if dataset[i].num_nodes > dataset[max_nodes_id].num_nodes:
        max_nodes_id = i
    if dataset[i].num_edges > dataset[max_edges_id].num_edges:
        max_edges_id = i
    num_nodes.append(dataset[i].num_nodes)
    num_edges.append(dataset[i].num_edges)
    for j in range(all_dataset[i].shape[0]):
        max_degree = np.max([max_degree, np.sum(all_dataset[i][j, :])])
        if max_degree == np.sum(all_dataset[i][j, :]):
            max_degree_id = i
print(max_degree)
lambda_coeff_data = np.min([LAMBDA_COEFF, 1.0 / (max_degree**2)])
print("lambda_coeff_data", lambda_coeff_data)
print(dataset[max_nodes_id])

"""Calculate graph features and compute time to create the gram matrix"""
start_gram_time = time.time()
num_random_walks = NB_RANDOM_WALKS
graph_features = []
for i in tqdm(range(len(all_dataset))):
    v = np.ones(all_dataset[i].shape[0]) / all_dataset[i].shape[0]
    w = np.ones(all_dataset[i].shape[0]) / all_dataset[i].shape[0]
    g = generate_approximate_graph_kernel_features(
        all_dataset[i],
        v,
        w,
        anchor_fraction=1.0,
        kernel_type="exponential",
        lambda_coeff=lambda_coeff_data,
        p_halt=P_HALT,
        nb_random_walks=num_random_walks,
    )
    graph_features.append(g)


"""calculate gram matrix"""
gram_matrix = np.zeros((len(all_dataset), len(all_dataset)))
for i in tqdm(range(len(all_dataset))):
    for j in range(i + 1):
        gram_matrix[i][j] = approximate_graph_kernel_value(
            graph_features[i]["latent_embedding"],
            graph_features[j]["latent_embedding"],
            nb_random_walks=num_random_walks,
        )
        gram_matrix[j][i] = gram_matrix[i][j]

total_gram_construction_time = time.time() - start_gram_time


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
acc = []

time_for_classification = time.time()
for train_index, test_index in skf.split(all_dataset, full_labels):

    train_labels = [full_labels[i] for i in train_index]
    test_labels = [full_labels[i] for i in test_index]
    # create train gram matrix
    train_mat = np.empty((len(train_index), len(train_index)))
    for i in range(len(train_index)):
        for j in range(len(train_index)):
            train_mat[i][j] = gram_matrix[train_index[i]][train_index[j]]

    svc = SVC(kernel="precomputed", verbose=True)

    svc.fit(train_mat, train_labels)

    kernel_test = np.empty((len(test_index), len(train_index)))
    for i in range(len(test_index)):
        for j in range(len(train_index)):
            kernel_test[i][j] = gram_mat[test_index[i]][train_index[j]]

    y_pred = svc.predict(kernel_test)
    accuracy = accuracy_score(test_labels, y_pred)
    acc.append(accuracy)
    print(f"accuracy: {accuracy}")

end_time_classification = time.time() - time_for_classification
print(
    f"Average accuracy : {np.average(acc)}, \nTime to create gram matrix : {total_gram_construction_time} \nTime for classification : {end_time_classification}"
)
