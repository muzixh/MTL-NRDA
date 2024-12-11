import numpy as np
import scipy.sparse as sp
import torch
from torch.utils import data
import pandas as pd
from texttable import Texttable

class ModelSaver:
    def __init__(self, args):
        self.args = args
        self.max_score = -np.inf
        self.no_improvement = 0
        self.weights = {'DLI': 0.4, 'DMI': 0.3, 'MLI': 0.3}
        self.patience = args.early_stopping
        self.min_improvement = 0.001
        self.epoch = 0

    def update_weights(self):

        if self.epoch < 50:
            self.weights = {'DLI': 0.4, 'DMI': 0.3, 'MLI': 0.3}
        elif 50 <= self.epoch < 100:
            self.weights = {'DLI': 0.35, 'DMI': 0.35, 'MLI': 0.3}
        else:
            self.weights = {'DLI': 1/3, 'DMI': 1/3, 'MLI': 1/3}

    def calculate_score(self, roc_val_DLI, roc_val_DMI, roc_val_MLI):
        return (self.weights['DLI'] * roc_val_DLI +
                self.weights['DMI'] * roc_val_DMI +
                self.weights['MLI'] * roc_val_MLI)

    def should_save_model(self, roc_val_DLI, roc_val_DMI, roc_val_MLI):
        self.epoch += 1
        self.update_weights()
        
        current_score = self.calculate_score(roc_val_DLI, roc_val_DMI, roc_val_MLI)
        improvement = current_score - self.max_score

        if improvement > self.min_improvement:
            self.max_score = current_score
            self.no_improvement = 0
            return True
        else:
            self.no_improvement += 1
            return False

    def should_stop(self):
        return self.no_improvement >= self.patience

def tab_printer(args):

    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_adjacency_matrix(A, I):

    A_tilde = A + 2 * I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(A, device):

    I = sp.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sp.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T).to(device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(device)
    return propagator

def features_to_sparse(features, device):

    index_1, index_2 = features.nonzero()
    values = [1.0]*len(index_1)
    node_count = features.shape[0]
    feature_count = features.shape[1]
    features = sp.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T).to(device)
    out_features["values"] = torch.FloatTensor(features.data).to(device)
    out_features["dimensions"] = features.shape
    return out_features
class Data_LRI(data.Dataset):
    def __init__(self, idx_map, labels, df):
        'Initialization'
        self.labels = labels
        self.idx_map = idx_map
        self.df = df

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label

        idx1 = self.idx_map[str(self.df.iloc[index].LEFT_ID)]
        idx2 = self.idx_map[self.df.iloc[index].RIGHT_ID]
        y = self.labels[index]
        return y, (idx1, idx2)

def load_data_link_prediction_LRI(path, network_type, inp, device):
    print('Loading LRI dataset...')
    path_up = f'./data/{network_type}'
    df_data = pd.read_csv(path + '/train.csv')
    df_drug_list = pd.read_csv(path_up + '/entity_list.csv')
    idx = df_drug_list['Entity_ID'].tolist()
    idx = np.array(idx)
    idx_map = {j: i for i, j in enumerate(idx)}

    df_data_t = df_data[df_data.label == 1]

    edges_unordered = df_data_t[['LEFT_ID', 'RIGHT_ID']].values
        
    if inp == 'node2vec':
        emb = pd.read_csv(path + '/lri.emb', skiprows=1, header=None, sep=' ').sort_values(by=[0]).set_index([0])
        new_index = [idx_map[idx] for idx in emb.index]
        emb = emb.reindex(new_index)

        for i in np.setdiff1d(np.arange(1276), emb.index.values):
            emb.loc[i] = (np.sum(emb.values, axis=0) / emb.values.shape[0])
        features = emb.sort_index().values
        features = normalize(features)

    elif inp == 'one_hot':
        features = np.eye(1276)

    features = features_to_sparse(features, device)

    edges = np.array(list(map(idx_map.get, edges_unordered.flatten()))).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = create_propagator_matrix(adj, device)

    return adj, features, idx_map

def load_data(args):

    data_path = f"./data/{args.network_type}/fold{args.fold_id}"

    if args.ratio:
        data_path = f"./data/{args.network_type}/{args.train_percent}/fold{args.fold_id}"

    if args.network_type == 'LRI':
        adj, features, idx_map = load_data_link_prediction_LRI(data_path, args.network_type, args.input_type, args.device)
        Data_class = Data_LRI
    return adj, features, idx_map, Data_class