import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score,roc_auc_score
from tqdm import tqdm
from dataset import CreateDataset
from torch.nn import BatchNorm1d, Sequential, ReLU
from conv import GATConv,GINConv
from torch_geometric.nn import MLP, global_add_pool

def compute_stats(dataset):
    features = []
    for data in dataset:
        features.append(data.x)
    all_features = torch.cat(features, dim=0)
    mean = all_features.mean(dim=0)
    std = all_features.std(dim=0)
    std[std == 0] = 1
    return mean, std

def normalization(dataset, mean, std):
    for data in dataset:
        data.x = (data.x - mean) / std
    return dataset

class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, n_classes):
        super(GNN, self).__init__()

        self.gcn1 = GATConv(in_dim, hidden_dim1, add_self_loops=False)
        self.gcn2 = GINConv(Sequential(
            MLP([hidden_dim1, hidden_dim2, hidden_dim2]),
            BatchNorm1d(hidden_dim2, hidden_dim2),
            ReLU()))
        self.gcn3 = GINConv(Sequential(
            MLP([hidden_dim2, hidden_dim2, hidden_dim2]),
            BatchNorm1d(hidden_dim2, hidden_dim2),
            ReLU()))
        self.BN = BatchNorm1d(hidden_dim1, hidden_dim1)
        self.fc = Sequential(
            nn.Linear(hidden_dim1 + hidden_dim2 *2, 32),
            nn.Linear(32, 24))
        self.classify = nn.Linear(24, n_classes)

    def forward(self, x, edge, batch, mode='default'):
        gcn1, att = self.gcn1(x, edge, return_attention_weights=True)
        gcn1 = F.relu(self.BN(gcn1))
        gcn2 = self.gcn2(gcn1, edge)
        gcn3 = self.gcn3(gcn2, edge)
        gcn = torch.cat([gcn1, gcn2, gcn3], dim=1)
        conv = global_add_pool(gcn, batch)
        out = F.relu(self.fc(conv))
        out = self.classify(out)
        if mode is not 'default':
            return out,att
        return out


class MultiGNN(torch.nn.Module):
    def __init__(self, in_dim_r, hidden_dim1_r, hidden_dim2_r,
                 in_dim_p, hidden_dim1_p, hidden_dim2_p, n_classes):
        super(MultiGNN, self).__init__()

        self.gcn1_r = GATConv(in_dim_r, hidden_dim1_r, add_self_loops=False)
        self.gcn1_p = GATConv(in_dim_p, hidden_dim1_p, add_self_loops=False)
        self.gcn2_r = GINConv(Sequential(
            MLP([hidden_dim1_r, hidden_dim2_r, hidden_dim2_r]),
            BatchNorm1d(hidden_dim2_r, hidden_dim2_r),
            ReLU()))
        self.gcn2_p = GINConv(Sequential(
            MLP([hidden_dim1_p, hidden_dim2_p, hidden_dim2_p]),
            BatchNorm1d(hidden_dim2_p, hidden_dim2_p),
            ReLU()))
        self.gcn3_r = GINConv(Sequential(
            MLP([hidden_dim2_r, hidden_dim2_r, hidden_dim2_r]),
            BatchNorm1d(hidden_dim2_r, hidden_dim2_r),
            ReLU()))
        self.gcn3_p = GINConv(Sequential(
            MLP([hidden_dim2_p, hidden_dim2_p, hidden_dim2_p]),
            BatchNorm1d(hidden_dim2_p, hidden_dim2_p),
            ReLU()))
        self.BN_r = BatchNorm1d(hidden_dim1_r, hidden_dim1_r)
        self.BN_p = BatchNorm1d(hidden_dim1_p, hidden_dim1_p)
        self.fc = Sequential(
            nn.Linear(hidden_dim1_r + hidden_dim2_r * 2 + hidden_dim1_p + hidden_dim2_p * 2, 32),
            nn.Linear(32, 24))
        self.classify = nn.Linear(24, n_classes)

    def forward(self, x_r, edge_r, x_p, edge_p, batch):
        gcn1_r = self.gcn1_r(x_r, edge_r, return_attention_weights=False)
        gcn1_r = F.relu(self.BN_r(gcn1_r))
        gcn2_r = self.gcn2_r(gcn1_r, edge_r)
        gcn3_r = self.gcn3_r(gcn2_r, edge_r)
        gcn_r = torch.cat([gcn1_r, gcn2_r, gcn3_r], dim=1)
        conv_r = global_add_pool(gcn_r, batch)
        gcn1_p = self.gcn1(x_p, edge_p, return_attention_weights=False)
        gcn1_p = F.relu(self.BN(gcn1_p))
        gcn2_p = self.gcn2(gcn1_p, edge_p)
        gcn3_p = self.gcn3(gcn2_p, edge_p)
        gcn_p = torch.cat([gcn1_p, gcn2_p, gcn3_p], dim=1)
        conv_p = global_add_pool(gcn_p, batch)
        conv = torch.cat([conv_r, conv_p], dim=1)
        out = F.relu(self.fc(conv))
        out = self.classify(out)
        return out

class Unimodal_Model(object):
    def __init__(self, label_path, root_path, node_path, edge_path, model_path, in_dim, hidden_dim1, hidden_dim2, epochs, epoch):
        self.in_dim = in_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.n_classes = 2
        self.learning_rate = 1e-3
        self.weight_decay = 5e-4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.label_path = label_path
        self.root_path = root_path
        self.node_path = node_path
        self.edge_path = edge_path
        self.model_path = model_path
        self.dataset = None
        self.train_set = None
        self.valid_set = None
        self.model = GNN(self.in_dims,self.hidden_dims1,self.hidden_dims2,self.n_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_batch_size = 32
        self.valid_batch_size = 1
        self.epochs = epochs
        self.epoch = epoch
        self.k = 4

    def normalization(dataset, mean, std):
        for data in dataset:
            data.x = (data.x - mean) / std
        return dataset

    def train(self):
        best_acc = 0
        best_auc = 0
        os.mkdir(os.path.join(self.root_path, 'dataset'))
        for i in range(1,self.k+1):
            os.mkdir(os.path.join(self.root_path, 'train'+str(i)))
            os.mkdir(os.path.join(self.root_path, 'valid'+str(i)))
        self.dataset = CreateDataset(os.path.join(self.root_path, 'dataset'), self.node_path, self.edge_path,
                                     pd.read_excel(self.label_path)['name'].tolist(),
                                     pd.read_excel(self.label_path)['label'].tolist())
        data_label = pd.read_excel(self.label_path).to_numpy()
        kf = KFold(n_splits=self.k)
        j = 0
        for train,valid in kf.split(data_label):
            j += 1
            self.train_set = CreateDataset(os.path.join(self.root_path, 'train'+str(j)), self.node_path, self.edge_path,
                                     data_label[train,0:1].tolist(),
                                     data_label[train,0:2].tolist())
            mean, std = self.compute_stats(self.train_set)
            train_loader = DataLoader(self.normalization(self.train_set, mean, std), batch_size=self.train_batch_size, shuffle=True)
            train_loss = 0
            train_nsample = 0

            n = 0
            labels = []
            preds = []
            self.model = self.model.to(self.device)
            self.model.train()
            t = tqdm(train_loader, desc=f'[train]epoch:{self.epoch}')
            for data in t:
                self.optimizer.zero_grad()
                data = data.to(self.device)
                out, _ = self.model(data.x, data.edge_index, data.batch, mode='train')
                loss = self.criterion(out, data.y)
                loss.backward()
                self.optimizer.step()
                train_nsample += len(out)
                train_loss += loss.item()
                n += 1
                pred = np.array(out.argmax(dim=1).cpu()).astype(int)
                for l in data.y.cpu():
                    labels.append(l)
                for p in pred:
                    preds.append(p)
            labels = np.array(labels)
            labels = labels.astype(int)
            preds = np.array(preds)
            preds = preds.astype(int)
            print(train_loss / train_nsample)
            print(accuracy_score(labels, preds))

            self.valid_set = CreateDataset(os.path.join(self.root_path, 'valid' + str(j)), self.node_path,
                                           self.edge_path,
                                           data_label[valid, 0:1].tolist(),
                                           data_label[valid, 0:2].tolist())
            self.model.eval()
            labels = []
            preds = []
            valid_loss = 0
            valid_nsample = 0
            valid_loader = DataLoader(self.valid_set, batch_size=self.valid_batch_size, shuffle=False)
            with torch.no_grad():
                for data in valid_loader:
                    data = data.to(self.device)
                    out, _ = self.model(data.x, data.edge_index, data.batch, mode='valid')
                    loss = self.criterion(out, data.y)
                    pred = out.argmax(dim=1).cpu()
                    valid_nsample += len(out)
                    valid += loss.item()
                    for l in data.y.cpu():
                        labels.append(l)
                    for p in pred:
                        preds.append(p)
            labels = np.array(labels)
            labels = labels.astype(int)
            preds = np.array(preds)
            preds = preds.astype(int)
            print(valid_loss / valid_nsample)
            acc = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, preds)
            if acc >= best_acc and auc >= best_auc:
                torch.save(self.model,os.path.join(self.model_path, "best_model_{:.2f}_{:.2f}.pt".format(acc, auc)))
                best_acc = acc
                best_auc = auc

class Multimodal_Model(object):
    def __init__(self, label_path,
                 root_path_r, node_path_r, edge_path_r,
                 root_path_p, node_path_p, edge_path_p,
                 model_path,
                 in_dim_r, hidden_dim1_r, hidden_dim2_r,
                 in_dim_p, hidden_dim1_p, hidden_dim2_p,
                 epochs, epoch):
        self.in_dim_r = in_dim_r
        self.in_dim_p = in_dim_p
        self.hidden_dim1_r = hidden_dim1_r
        self.hidden_dim1_p = hidden_dim1_p
        self.hidden_dim2_r = hidden_dim2_r
        self.hidden_dim2_p = hidden_dim2_p
        self.n_classes = 2
        self.learning_rate = 1e-3
        self.weight_decay = 5e-4
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.label_path = label_path
        self.root_path_r = root_path_r
        self.node_path_r = node_path_r
        self.edge_path_r = edge_path_r
        self.root_path_p = root_path_p
        self.node_path_p = node_path_p
        self.edge_path_p = edge_path_p
        self.model_path = model_path
        self.dataset_r = None
        self.dataset_p = None
        self.train_set_r = None
        self.train_set_p = None
        self.valid_set_r = None
        self.valid_set_p = None
        self.model = MultiGNN(self.in_dim_r,self.hidden_dim1_r,self.hidden_dim2_r,
                              self.in_dim_p,self.hidden_dim1_p,self.hidden_dim2_p,
                              self.n_classes)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.train_batch_size = 32
        self.valid_batch_size = 1
        self.epochs = epochs
        self.epoch = epoch
        self.k = 4

    def train(self):
        g = torch.Generator()
        best_acc = 0
        best_auc = 0
        self.dataset_r = CreateDataset(os.path.join(self.root_path_r, 'dataset'), self.node_path_r, self.edge_path_r,
                                     pd.read_excel(self.label_path)['name'].tolist(),
                                     pd.read_excel(self.label_path)['label'].tolist())
        self.dataset_p = CreateDataset(os.path.join(self.root_path_p, 'dataset'), self.node_path_p, self.edge_path_p,
                                     pd.read_excel(self.label_path)['name'].tolist(),
                                     pd.read_excel(self.label_path)['label'].tolist())
        data_label = pd.read_excel(self.label_path).to_numpy()
        kf = KFold(n_splits=self.k)
        j = 0
        for train,valid in kf.split(data_label):
            j += 1
            g.manual_seed(j)
            self.train_set_r = CreateDataset(os.path.join(self.root_path_r, 'train'+str(j)), self.node_path_r,
                                             self.edge_path_r,
                                             data_label[train,0:1].tolist(),
                                             data_label[train,0:2].tolist())
            self.train_set_p = CreateDataset(os.path.join(self.root_path_p, 'train' + str(j)), self.node_path_p,
                                             self.edge_path_p,
                                             data_label[train, 0:1].tolist(),
                                             data_label[train, 0:2].tolist())
            mean_r, std_r = compute_stats(self.train_set_r)
            mean_p, std_p = compute_stats(self.train_set_p)
            train_loader_r = DataLoader(normalization(self.train_set_r, mean_r, std_r), batch_size=self.train_batch_size, shuffle=True, generator=g)
            train_loader_p = DataLoader(normalization(self.train_set_p, mean_p, std_p), batch_size=self.train_batch_size, shuffle=True, generator=g)
            train_loss = 0
            train_nsample = 0

            n = 0
            labels = []
            preds = []
            self.model = self.model.to(self.device)
            self.model.train()
            t_r = tqdm(train_loader_r, desc=f'[train]epoch:{self.epoch}')
            t_p = tqdm(train_loader_p, desc=f'[train]epoch:{self.epoch}')
            for data_r,data_p in zip(t_r,t_p):
                self.optimizer.zero_grad()
                data_r = data_r.to(self.device)
                data_p = data_p.to(self.device)
                out = self.model(data_r.x, data_r.edge_index,
                                 data_p.x, data_p.edge_index,
                                 data_r.batch)
                loss = self.criterion(out, data_r.y)
                loss.backward()
                self.optimizer.step()
                train_nsample += len(out)
                train_loss += loss.item()
                n += 1
                pred = np.array(out.argmax(dim=1).cpu()).astype(int)
                for l in data_r.y.cpu():
                    labels.append(l)
                for p in pred:
                    preds.append(p)
            labels = np.array(labels)
            labels = labels.astype(int)
            preds = np.array(preds)
            preds = preds.astype(int)
            print(train_loss / train_nsample)
            print(accuracy_score(labels, preds))

            self.valid_set_r = CreateDataset(os.path.join(self.root_path_r, 'valid' + str(j)), self.node_path_r,
                                           self.edge_path_r,
                                           data_label[valid, 0:1].tolist(),
                                           data_label[valid, 0:2].tolist())
            self.valid_set_p = CreateDataset(os.path.join(self.root_path_p, 'valid' + str(j)), self.node_path_p,
                                             self.edge_path_p,
                                             data_label[valid, 0:1].tolist(),
                                             data_label[valid, 0:2].tolist())
            self.model.eval()
            labels = []
            preds = []
            valid_loss = 0
            valid_nsample = 0
            valid_loader_r = DataLoader(self.valid_set_r, batch_size=self.valid_batch_size, shuffle=False)
            valid_loader_p = DataLoader(self.valid_set_p, batch_size=self.valid_batch_size, shuffle=False)
            with torch.no_grad():
                v_r = tqdm(valid_loader_r, desc=f'[valid]epoch:{self.epoch}')
                v_p = tqdm(valid_loader_p, desc=f'[valid]epoch:{self.epoch}')
                for data_r,data_p in zip(v_r, v_p):
                    data_r = data_r.to(self.device)
                    data_p = data_p.to(self.device)
                    out = self.model(data_r.x, data_p.edge_index,
                                     data_p.x, data_p.edge_index,
                                     data_r.batch)
                    loss = self.criterion(out, data_r.y)
                    pred = out.argmax(dim=1).cpu()
                    valid_nsample += len(out)
                    valid += loss.item()
                    for l in data_r.y.cpu():
                        labels.append(l)
                    for p in pred:
                        preds.append(p)
            labels = np.array(labels)
            labels = labels.astype(int)
            preds = np.array(preds)
            preds = preds.astype(int)
            print(valid_loss / valid_nsample)
            acc = accuracy_score(labels, preds)
            auc = roc_auc_score(labels, preds)
            if acc >= best_acc and auc >= best_auc:
                torch.save(self.model,os.path.join(self.model_path, "best_model_{:.2f}_{:.2f}.pt".format(acc, auc)))
                best_acc = acc
                best_auc = auc

