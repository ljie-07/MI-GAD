import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from model import GAD
import torch_geometric.utils as utils
from utils import *
from args import parameter_parser
from Dataloader import load_mat
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, auc, precision_score, precision_recall_curve
import torch.nn.functional as F
import math
import random
import os
from tqdm import tqdm
import json
from utils import *



def local_affinity(feature_normalized, adj_matrix):

    if adj_matrix.is_sparse:
        row_sum = torch.sparse.sum(adj_matrix, dim=1).to_dense()
    else:
        row_sum = adj_matrix.sum(dim=1)

    row_sum = torch.clamp(row_sum, min=1e-30)

    if args.dataset == 'Elliptic-all' or args.dataset == 'dgraphfin':
        feature_normalized = feature_normalized
    else:
        feature_normalized = F.normalize(feature_normalized, p=2, dim=1)

    if adj_matrix.is_sparse:
        agg_features = torch.sparse.mm(adj_matrix, feature_normalized)
    else:
        agg_features = torch.matmul(adj_matrix, feature_normalized)

    message = (feature_normalized * agg_features).sum(dim=1) / row_sum


    return message


def get_loss(z,test_id):
    z = F.normalize(z, p=2, dim=-1)
    if args.dataset == 'Elliptic-all' or args.dataset =='dgraphfin':
        z_centered = z[test_id].mean(dim=0)
    else:
        z_centered = z.mean(dim=0)
    z_centered = z - z_centered
    loss_per_sample = z_centered.pow(2).sum(dim=1)
    total_loss = loss_per_sample.mean()

    return loss_per_sample, total_loss




args = parameter_parser()
print(args)

# Load and preprocess data
adj, features, label, adj_label, test_id = load_mat(args.dataset)

if args.dataset in ['Reddit']:
    features = (features - features.mean(0)) / (features.std(0) + 1e-30)


features = torch.FloatTensor(features)
nb_nodes = features.shape[0]
seeds = [i for i in range(args.runs)]
all_auc = []
all_auprc = []

for run in range(args.runs):
    seed = seeds[run]
    print('\n# Run:{}'.format(run), flush=True)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    loss_values = []
    aucc = []

    cnt_wait = 0
    best = 1e9
    best_t = 0
    patience = args.patience
    max_score = 0.
    b_xent = nn.BCEWithLogitsLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()
    feat_size = features.size(1)

    model = GAD(feat_size=features.size(1), hidden_size=args.hidden_dim, dropout=args.dropout)
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device(args.device)
        adj = adj.to(device)
        adj_label = adj_label.to(device)
        features = features.to(device)
        model = model.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    time_train = time.time()

    for epoch in range(args.epoch):
        model.train()
        optimiser.zero_grad()

        # Forward pass
        node_embed = model(features)

        sc1 = local_affinity(node_embed, adj_label)
        dis_adj = negative_sampling(adj_label)
        sc2 = local_affinity(node_embed, dis_adj)

        g_loss, gl_loss = get_loss(node_embed,test_id)

        pred = torch.cat([sc1, sc2], dim=0)

        lbl = torch.cat([torch.ones(nb_nodes), torch.zeros(nb_nodes)], dim=0).to(device)

        loss_affinity = b_xent(pred, lbl).mean()





        score1 = -sc1
        score1 = score1.detach().cpu().numpy()
        score2 = g_loss.detach().cpu().numpy()

        scaler1 = MinMaxScaler()
        scaler2 = MinMaxScaler()

        score1 = scaler1.fit_transform(score1.reshape(-1, 1)).reshape(-1)
        score2 = scaler2.fit_transform(score2.reshape(-1, 1)).reshape(-1)

        score = score1 * args.alpha + score2 * args.beta


        auc1 = roc_auc_score(label, score)
        auprc = calculate_auprc(label, score)

        l = loss_affinity * args.alpha + gl_loss * args.beta




        if l < best and epoch > args.epoch//2:
            best = l
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break

        l.backward()
        optimiser.step()

        loss_tu = l.item()
        loss_values.append(loss_tu)
        print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(l.item()), "auc:", auc1)

    time_train = time.time() - time_train
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('best_model.pth'))

    multi_round_ano_score = np.zeros((args.tests, nb_nodes))



    for test in range(args.tests):
        with torch.no_grad():


            node_embed= model(features)

            mem_test = torch.cuda.max_memory_allocated()

            sc1 = local_affinity(node_embed, adj_label)

            g_loss, gl_loss = get_loss(node_embed, test_id)
            score1 = -sc1
            score1 = score1.detach().cpu().numpy()
            score2 = g_loss.detach().cpu().numpy()

            scaler1 = MinMaxScaler()
            scaler2 = MinMaxScaler()
            score1 = scaler1.fit_transform(score1.reshape(-1, 1)).reshape(-1)
            score2 = scaler2.fit_transform(score2.reshape(-1, 1)).reshape(-1)

            score = score1*args.alpha + score2 * args.beta


            multi_round_ano_score[test] = score

            if args.dataset == 'dgraphfin' or args.dataset =='Elliptic-all':
                auc_score = roc_auc_score(label[test_id], score[test_id])
            else:
                auc_score = roc_auc_score(label, score)


            aucc.append(auc_score)
            if auc_score > max_score:
                max_score = auc_score

            print("Epoch:", '%04d' % (test), 'Auc:', auc_score, 'Best_Auc:', max_score)

    ano_score_final = np.mean(multi_round_ano_score, axis=0)
    auc_score = roc_auc_score(label, ano_score_final)
    auprc = calculate_auprc(label, ano_score_final)

    print("auprc:", auprc)
    print("auc:", auc_score)

    all_auc.append(auc_score)
    all_auprc.append(auprc)

    del sc1, sc2, pred, lbl, model
    torch.cuda.empty_cache()

print('\n==============================')
print(all_auc)
print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)*100), 'FINAL TESTING AUC std:{:.4f}'.format(np.std(all_auc)*100))
print('{:.2f} ({:.2f})'.format(np.mean(all_auc)*100, np.std(all_auc)*100))
print(all_auprc)
print('FINAL TESTING AUPRC:{:.4f}'.format(np.mean(all_auprc)*100),
      'FINAL TESTING AUPRC std:{:.4f}'.format(np.std(all_auprc)*100))
print('{:.2f} ({:.2f})'.format(np.mean(all_auprc)*100, np.std(all_auprc)*100))
print('==============================')


