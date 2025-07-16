from sklearn.metrics.pairwise import cosine_similarity
import os
import pandas as pd

if __name__ == '__main__':
    radio_feature_path = 'E:\\dce\\test\\node'
    radio_edge_path = 'E:\\dce\\test\\edge'
    radio_cosine_path =  'E:\\dce\\test\\cosine'
    patho_feature_path = 'E:\\wsi\\test\\node'
    patho_edge_path = 'E:\\wsi\\test\\edge'
    patho_cosine_path = 'E:\\wsi\\test\\cosine'
    label = pd.read_excel('E:\\test_label.xlsx')
    for name in label['name']:
        df_r = pd.DataFrame()
        radio_feature = os.path.join(radio_feature_path, name+'.xlsx')
        radio_edge = os.path.join(radio_edge_path, name+'.xlsx')
        for length in range(len(radio_edge)):
            node1 = radio_edge.iloc[length].tolist()[0]
            node2 = radio_edge.iloc[length].tolist()[1]
            feature = pd.DataFrame(cosine_similarity(radio_feature.iloc[node1].to_numpy(),radio_feature.iloc[node2].to_numpy()))
            df_r = pd.concat([df_r, feature], axis=0)
        df_r.columns = ['Cosine_similarity']
        df_r.to_excel(os.path.join(radio_cosine_path, name+'.xlsx'))

        df_p = pd.DataFrame()
        patho_feature = os.path.join(patho_feature_path, name + '.xlsx')
        patho_edge = os.path.join(patho_edge_path, name + '.xlsx')
        for length in range(len(radio_edge)):
            node1 = patho_edge.iloc[length].tolist()[0]
            node2 = patho_edge.iloc[length].tolist()[1]
            feature = pd.DataFrame(
                cosine_similarity(patho_feature.iloc[node1].to_numpy(), patho_feature.iloc[node2].to_numpy()))
            df_p = pd.concat([df_p, feature], axis=0)
        df_p.columns = ['Cosine_similarity']
        df_p.to_excel(os.path.join(patho_cosine_path, name + '.xlsx'))


