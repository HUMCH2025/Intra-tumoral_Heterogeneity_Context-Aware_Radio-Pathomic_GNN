import torch
import torch.nn.functional as F

from torch_geometric.utils import to_undirected
from torch_geometric.data import Data,InMemoryDataset

import pandas as pd

class CreateDataset(InMemoryDataset):
    def __init__(self, root, v_path, e_path, x, y, transform=None, pre_transform=None):
        self.v_path = v_path
        self.e_path = e_path
        self.x = x
        self.y = y
        self.length = 0
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        print(self.data)

    def raw_file_names(self):
        return []

    def processed_file_names(self):
        return ['datas.pt']

    def download(self):
        pass

    def process(self):
        n = 0
        data_list = []
        for name_path in self.x:
            edge_index = torch.tensor(pd.read_excel(self.e_path + name_path + ".xlsx").values, dtype=torch.int).t()
            edge_index = to_undirected(edge_index)
            X = torch.tensor(pd.read_excel(self.v_path + name_path + ".xlsx").values, dtype=torch.float)
            X = F.normalize(X, dim=1)
            Y = torch.tensor(self.y[n], dtype=torch.long)
            N_v = torch.tensor(X.shape[0], dtype=torch.int64)
            N_e = torch.tensor(edge_index.shape[1], dtype=torch.int64)
            n += 1

            data = Data(x=X, edge_index=edge_index, y=Y, pos=[N_v, N_e])
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        self.length = n

