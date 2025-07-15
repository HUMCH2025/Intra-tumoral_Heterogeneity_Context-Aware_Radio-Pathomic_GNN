import os
import math
import pandas as pd
import nibabel as nib
import numpy as np
import SLICProcessor
import Radiomic_Feature
import MergeProcessor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

def num_to_k(num):
    if  0 < num <= 2000:
        k = math.ceil(num / 100)
    elif 2000 < num <= 5000:
        k = math.ceil((num - 2000) / 300) + 20
    elif 5000 < num <= 10000:
        k = math.ceil((num - 5000) / 500) + 30
    elif 10000 < num <= 100000:
        k = math.ceil((num - 10000) / 1000) + 40
    else:
        k = math.ceil((num - 100000) / 2000) + 130
    return k

def main():
    dce_path = 'E:\\dce_resize\\test'
    dce_list = ['curve_shape_index.nii.gz','Enhancement at first postcontrast time point.nii.gz',
                'maximum_enhancement.nii.gz','SER.nii.gz',
                'time_to_peak.nii.gz','uptake_rate.nii.gz',
                'washout_rate.nii.gz']
    mask_name = 'DCE-label.nii.gz'

    for path in os.listdir(dce_path):
        df = pd.DataFrame()
        num = 0
        List = []
        for dce_name in dce_list:
            image_path = os.path.join(os.path.join(dce_path, path), dce_name)
            mask_path = os.path.join(os.path.join(dce_path, path), mask_name)
            num += 1
            image_obj = nib.load(image_path)
            mask_obj = nib.load(mask_path)
            image_data = image_obj.get_fdata()
            mask_data = mask_obj.get_fdata()
            site = np.nonzero(mask_data)
            d_list = []
            if num == 1:
                new_array = np.zeros((site[0].max()-site[0].min()+1,site[1].max()-site[1].min()+1,site[2].max()-site[2].min()+1,7))
                new_array = np.full_like(new_array, -10)
                for i in range(len(site[0])):
                    List.append([site[0][i], site[1][i], site[2][i]])
            for num in range(len(List)):
                a,b,c = List[num][0],List[num][1],List[num][2]
                d = image_data[a,b,c]
                d_list.append(d)
            df_add = pd.DataFrame.from_dict(d_list)
            df = pd.concat([df,df_add],axis=1)

        df = PCA(n_components=3).fit_transform(df)
        feature = MinMaxScaler().fit_transform(df)
        new_mask = mask_data[site[0].min():site[0].max()+1,site[1].min():site[1].max()+1,site[2].min():site[2].max()+1]
        n = 0
        pos_list = []
        arr_list = []
        for i in range(len(new_mask)):
            for j in range(len(new_mask[0])):
                for k in range(len(new_mask[0][0])):
                    if new_mask[i][j][k] == 1:
                        new_array[i][j][k] = feature[n]
                        n += 1
                        pos_list.append([i,j,k])
                        arr_list.append([new_array[i][j][k][0],new_array[i][j][k][1],new_array[i][j][k][2]])
        arrlist = np.array(arr_list)
        poslist = np.array(pos_list)
        feature = []
        position = []
        edge = []
        al = np.nonzero(arrlist != -10)
        al = np.unique(al[0]).tolist()
        for i in al:
            feature.append(arrlist[i].tolist())
            position.append(poslist[i].tolist())
        count = num_to_k(int(len(List)))
        label = SLICProcessor(new_array, count+10, 0.1).iterates()
        label = MergeProcessor(label, new_mask, count).merge()
        i = np.nonzero(new_mask)
        for j in range(len(i[0])):
            for h in range(i[0][j]-1,i[0][j]+2):
                if h < 0 or h > i[0].max():
                    continue
                for w in range(i[1][j]-1,i[1][j]+2):
                    if w < 0 or w > i[1].max():
                        continue
                    for d in range(i[2][j]-1,i[2][j]+2):
                        if d < 0 or d > i[2].max():
                            continue
                        if label[h][w][d] > label[i[0][j]][i[1][j]][i[2][j]] and (new_array[h][w][d] != -10).all():
                            edge.append((label[i[0][j]][i[1][j]][i[2][j]]-1,label[h][w][d]-1))
        new_edge = []
        for e in edge:
            if e not in new_edge:
                new_edge.append(e)
        image_component1 = np.zeros_like(new_mask)
        image_component2 = np.zeros_like(new_mask)
        image_component3 = np.zeros_like(new_mask)
        for i in range(len(site[0])):
            image_component1[site[0][i], site[1][i], site[2][i]] = feature[:, 0][i]
            image_component2[site[0][i], site[1][i], site[2][i]] = feature[:, 1][i]
            image_component3[site[0][i], site[1][i], site[2][i]] = feature[:, 2][i]

        all_data = pd.DataFrame()
        for i in np.unique(label):
            mask = np.where(new_mask == i, 1, 0)
            data = Radiomic_Feature(image_component1, image_component2, image_component3, mask).get_features()
            all_data = pd.concat([all_data, data], axis=0)
            all_data.columns = data.columns

        new_edge = pd.DataFrame(new_edge)
        new_edge.columns = ['v1','v2']
        all_data.to_excel("E:\\dce\\test\\node\\"+path.split('\\')[-1]+".xlsx",index=False)
        new_edge.to_excel("E:\\dce\\test\\edge\\"+path.split('\\')[-1]+".xlsx",index=False)

if __name__ == '__main__':
    main()