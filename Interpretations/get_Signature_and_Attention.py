from torch_geometric.explain import CaptumExplainer
from torch_geometric.explain import Explainer
from Graph_Conv.models import GNN,MultiGNN
from Graph_Conv.dataset import CreateDataset
import os
import torch
import pandas as pd

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

if __name__ == '__main__':
    radio_model = GNN(135,32,24,2)
    patho_model = GNN(186,36,24,2)
    radio_patho_model = MultiGNN(135,32,24,186,32,24,2)
    radio_model = torch.load('E:\\dce\\model\\best_model_0.87_0.89.pt')
    patho_model = torch.load('E:\\wsi\\model\\best_model_0.85_0.87.pt')
    radio_patho_model = torch.load('E:\\model\\best_model_0.92_0.95.pt')

    train_radio_dataset = CreateDataset('E:\\dce\\root\\train\\dataset\\', 'E:\\dce\\train\\node', 'E:\\dce\\train\\edge',
                                        pd.read_excel('E:\\train_label.xlsx')['name'].tolist(),
                                        pd.read_excel('E:\\train_label.xlsx')['label'].tolist())
    train_patho_dataset = CreateDataset('E:\\wsi\\root\\train\\dataset', 'E:\\wsi\\train\\node', 'E:\\wsi\\train\\edge',
                                        pd.read_excel('E:\\train_label.xlsx')['name'].tolist(),
                                        pd.read_excel('E:\\train_label.xlsx')['label'].tolist())
    mean_r, std_r = compute_stats(train_radio_dataset)
    mean_p, std_p = compute_stats(train_patho_dataset)
    radio_dataset = CreateDataset('E:\\dce\\root\\test', 'E:\\dce\\test\\node', 'E:\\dce\\test\\edge',
                                  pd.read_excel('E:\\test_label.xlsx')['name'].tolist(),
                                  pd.read_excel('E:\\test_label.xlsx')['label'].tolist())
    patho_dataset = CreateDataset('E:\\wsi\\root\\test', 'E:\\wsi\\test\\node', 'E:\\wsi\\test\\edge',
                                  pd.read_excel('E:\\test_label.xlsx')['name'].tolist(),
                                  pd.read_excel('E:\\test_label.xlsx')['label'].tolist())
    norm_radio = normalization(radio_dataset, mean_r, std_r)
    norm_patho = normalization(patho_dataset, mean_p, std_p)

    df = pd.DataFrame()
    for length in len(radio_dataset):
        name = pd.read_excel('E:\\test_label.xlsx')['name'][length]
        radio_data = norm_radio[length].to(device)
        patho_data = norm_patho[length].to(device)
        radio_sig, radio_att = radio_model(radio_data.x, radio_data.edge_index, torch.tensor([0]*len(radio_data.x), device=device),mode='test')
        patho_sig, patho_att = patho_model(patho_data.x, patho_data.edge_index, torch.tensor([0]*len(patho_data.x), device=device),mode='test')
        radio_patho_sig = radio_patho_model(radio_data.x, radio_data.edge_index, patho_data.x, patho_data.edge_index, torch.tensor([0]*len(radio_data.x), device=device))
        radio_att = pd.DataFrame(radio_att.detach().numpy())
        patho_att = pd.DataFrame(patho_att.detach().numpy())
        radio_att.to_excel(os.path.join('E:\\dce\\test\\attention',name+'.xlsx'),index=False)
        patho_att.to_excel(os.path.join('E:\\wsi\\test\\attention', name + '.xlsx'), index=False)
        data = pd.DataFrame([name,radio_sig,patho_sig,radio_patho_sig])
        df = pd.concat([df, data],axis=0)
    df.columns = ['Name','Radiomic_Signature','Pathomic_Signature','Radio-pathomic_Signature']
    df.to_excel('E:\\test\\Signature.xlsx',index=False)
