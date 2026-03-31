import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
import copy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn as nn
import scipy.io as sio
import random
from sklearn.metrics import precision_recall_curve,auc
# import dgl
import math
from sklearn import manifold
from sklearn.neighbors import kneighbors_graph
from args import parameter_parser
import torch.nn.functional as F
from torch.sparse import mm
import os.path as osp
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
args = parameter_parser()
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman'] + rcParams['font.serif']

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize(mx):
    """Row-normalize sparse matrix"""

    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()

    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    # sp_tuple = sparse_to_tuple(features)
    if sp.issparse(features):
        features = features.todense()
    return features






def load_adj(features):
    adj = kneighbors_graph(features, 10)

    adj = sp.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj

def calculate_auprc(y_true, y_scores):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auprc = auc(recall, precision)
    return auprc





def negative_sampling(raw_adj):

    device = raw_adj.device  # 获取 raw_adj 的设备
    adj = raw_adj.coalesce()  # 确保索引是唯一且有序的
    indices = adj.indices()
    N = raw_adj.size(0)
    num_positive = indices.size(1)# 正样本数量


    i = indices[0]
    j = indices[1]
    existing_edges = (i * N + j).unique()


    negative_edges = torch.empty((2, num_positive), dtype=torch.long, device=device)


    neg_sample_idx = 0
    while neg_sample_idx < num_positive:
        batch_size = num_positive - neg_sample_idx
        sampled_i = torch.randint(0, N, (batch_size,), dtype=torch.long, device=device)
        sampled_j = torch.randint(0, N, (batch_size,), dtype=torch.long, device=device)


        mask = sampled_i < sampled_j
        sampled_i = sampled_i[mask]
        sampled_j = sampled_j[mask]


        sampled_encoded = sampled_i * N + sampled_j


        positions = torch.searchsorted(existing_edges, sampled_encoded)
        mask_not_exist = (positions >= existing_edges.size(0)) | (existing_edges[positions] != sampled_encoded)


        sampled_i = sampled_i[mask_not_exist]
        sampled_j = sampled_j[mask_not_exist]


        num_new_samples = sampled_i.size(0)
        if num_new_samples > 0:
            negative_edges[:, neg_sample_idx:neg_sample_idx + num_new_samples] = torch.stack([sampled_i, sampled_j], dim=0)
            neg_sample_idx += num_new_samples


    neg_values = torch.ones(neg_sample_idx, dtype=raw_adj.dtype, device=device)
    neg_adj = torch.sparse_coo_tensor(negative_edges[:, :neg_sample_idx], neg_values, raw_adj.size()).coalesce()

    return neg_adj



# def negative_sampling(raw_adj):
#
#
#
#     device = raw_adj.device  # 获取 raw_adj 的设备
#     adj = raw_adj.coalesce()  # 确保索引是唯一且有序的
#     indices = adj.indices()
#     N = raw_adj.size(0)
#     num_positive = indices.size(1)  # 正样本数量
#
#
#     i = indices[0]
#     j = indices[1]
#     existing_edges = (i * N + j).unique()
#
#
#     existing_edges_mask = torch.zeros(N * N, dtype=torch.bool, device=device)
#     existing_edges_mask[existing_edges] = True
#
# Code for reviewers






