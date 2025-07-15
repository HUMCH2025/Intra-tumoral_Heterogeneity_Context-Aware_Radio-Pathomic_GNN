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
    radio_explainer = Explainer(model=radio_model,
                                algorithm=CaptumExplainer('IntegratedGradients',n_steps=200),
                                explanation_type='model',
                                node_mask_type='attributes',
                                edge_mask_type=None,
                                model_config=dict(mode='multiclass_classification',
                                                  task_level='graph',
                                                  return_type='probs'),
                                )
    patho_explainer = Explainer(model=patho_model,
                                algorithm=CaptumExplainer('IntegratedGradients', n_steps=200),
                                explanation_type='model',
                                node_mask_type='attributes',
                                edge_mask_type=None,
                                model_config=dict(mode='multiclass_classification',
                                                  task_level='graph',
                                                  return_type='probs'),
                                )
    radio_patho_explainer = Explainer(model=radio_patho_model,
                                      algorithm=CaptumExplainer('IntegratedGradients', n_steps=200),
                                      explanation_type='model',
                                      node_mask_type='attributes',
                                      edge_mask_type=None,
                                      model_config=dict(mode='multiclass_classification',
                                                        task_level='graph',
                                                        return_type='probs'),
                                      )
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

    df_all = pd.DataFrame()
    for length in len(radio_dataset):
        df = pd.DataFrame()
        List = []
        name = pd.read_excel('E:\\test_label.xlsx')['name'][length]
        radio_data = norm_radio[length].to(device)
        patho_data = norm_patho[length].to(device)
        radio_explaination = radio_explainer(x=radio_data.x,edge_index=radio_data.edge_index,batch=torch.tensor([0]*len(radio_data.x), device=device))
        patho_explaination = patho_explainer(x=patho_data.x,edge_index=patho_data.edge_index,batch=torch.tensor([0]*len(patho_data.x), device=device))
        radio_node_ig = torch.sum(radio_explaination.node_mask, dim=1)
        patho_node_ig = torch.sum(patho_explaination.node_mask, dim=1)
        data = pd.concat([pd.DataFrame(radio_node_ig),pd.DataFrame(patho_node_ig)],axis=1)
        for l in range(len(patho_node_ig)):
            List.append(name+'_'+str(l))
        dt = pd.DataFrame(List)
        dataframe = pd.concat([dt,patho_node_ig],axis=1)
        df = pd.concat([df,dataframe],axis=0)
        df.columns = ['name_node', 'IG_Value']
        feature = pd.read_excel("E:\\wsi\\test\\node\\"+name+".xlsx")
        df_all = pd.concat([df_all, pd.concat([df, feature],axis=1)],axis=0)
        data.columns = ['Radiomic_IG','Pathomic_IG']
        data.to_excel(os.path.join('E:\\test\\IG_Value',name+'.xlsx'),index=False)

    df_all.to_excel(os.path.join('E:\\test\\IG_Feature.xlsx'), index=False)

