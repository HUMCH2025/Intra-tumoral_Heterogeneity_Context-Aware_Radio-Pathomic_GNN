import math
import os
import cv2
import numpy as np
import pandas as pd
import SLICProcessor
import MergeProcessor
import Subregion_feature_extraction


def get_superpixels(tumor_path, immune_path, fibroblast_path, other_path, N):
    dst1 = cv2.imread(tumor_path)
    dst2 = cv2.imread(immune_path)
    dst3 = cv2.imread(fibroblast_path)
    dst4 = cv2.imread(other_path)
    h, w = dst1.shape
    matrix = np.zeros((h, w, 4))
    for i in range(math.ceil(h / N)):
        for j in range(math.ceil(w / N)):
            if (i + 1) * N > h and (j+1) * N > w:
                roi1 = dst1[i * N:h, j * N:w].copy()
                roi2 = dst2[i * N:h, j * N:w].copy()
                roi3 = dst3[i * N:h, j * N:w].copy()
                roi4 = dst4[i * N:h, j * N:w].copy()

            elif (i + 1) * N > h:
                roi1 = dst1[i * N:h, j * N:(j + 1) * N].copy()
                roi2 = dst2[i * N:h, j * N:(j + 1) * N].copy()
                roi3 = dst3[i * N:h, j * N:(j + 1) * N].copy()
                roi4 = dst4[i * N:h, j * N:(j + 1) * N].copy()

            elif (j + 1) * N > w:
                roi1 = dst1[i * N:(i + 1) * N, j * N:w].copy()
                roi2 = dst2[i * N:(i + 1) * N, j * N:w].copy()
                roi3 = dst3[i * N:(i + 1) * N, j * N:w].copy()
                roi4 = dst4[i * N:(i + 1) * N, j * N:w].copy()

            else:
                roi1 = dst1[i * N:(i + 1) * N, j * N:(j + 1) * N].copy()
                roi2 = dst2[i * N:(i + 1) * N, j * N:(j + 1) * N].copy()
                roi3 = dst3[i * N:(i + 1) * N, j * N:(j + 1) * N].copy()
                roi4 = dst4[i * N:(i + 1) * N, j * N:(j + 1) * N].copy()

            contours1, hierarchy1 = cv2.findContours(roi1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours2, hierarchy2 = cv2.findContours(roi2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours3, hierarchy3 = cv2.findContours(roi3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours4, hierarchy4 = cv2.findContours(roi4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            tumor_num = 0
            immune_num = 0
            other_num = 0
            fibro_num = 0
            all_num = 0
            for a in range(len(contours1)):
                tumor_num += 1
                all_num += 1
            for b in range(len(contours2)):
                immune_num += 1
                all_num += 1
            for c in range(len(contours3)):
                fibro_num += 1
                all_num += 1
            for d in range(len(contours4)):
                other_num += 1
                all_num += 1
            tumor_num = tumor_num / all_num * 100
            immune_num = immune_num / all_num * 100
            fibro_num =  fibro_num / all_num * 100
            other_num = 100 - tumor_num - immune_num - fibro_num
            matrix[i][j][0] = tumor_num
            matrix[i][j][1] = immune_num
            matrix[i][j][2] = fibro_num
            matrix[i][j][3] = other_num

    mask = np.where(matrix>0, 1, 0)
    merge = SLICProcessor(matrix, 60, 0.01).iterates()
    merge = MergeProcessor(merge, mask, 51)
    label = np.zeros((h,w))
    for i in range(h):
        for j in range(w):
            if (i + 1) * N > h and (j + 1) * N > w:
                label[i * N:h, j * N:w] = merge[i, j]
            if (i + 1) * N > h:
                label[i * N:h, j * N:(j + 1) * N] = merge[i, j]
            if (j + 1) * N > w:
                label[i * N:(i + 1) * N, j * N:w] = merge[i, j]
            else:
                label[i * N:(i + 1) * N, j * N:(j + 1) * N] = merge[i, j]
    return label, matrix

def get_edge(label):
    edge = []
    for l in np.unique(label):
        mask = np.where(label == l, 1, 0)
        for nx, ny in zip(np.nonzero(mask)[0], np.nonzero(mask)[1]):
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                x = dx + nx
                y = dy + ny
                if label[nx][ny] == l and label[x][y] != l and label[x][y] != 0:
                    edge.append([label[nx][ny]-1, label[x][y]-1])

    new_edge = []
    for e in edge:
        if e not in new_edge:
            new_edge.append(e)
    new_edge = pd.DataFrame(new_edge)
    new_edge.columns = ['v1', 'v2']
    return new_edge

def main():
    wsi_path = 'E:\\wsi\\'

    for path in os.listdir(wsi_path):
        df = pd.DataFrame()
        patient_path = os.path.join(wsi_path, path)
        label, matrix = get_superpixels(os.path.join(patient_path, 'tumor_mask.tif'), os.path.join(patient_path, 'immune_mask.tif'),
                                        os.path.join(patient_path, 'fibroblast_mask.tif'), os.path.join(patient_path, 'other_mask.tif'),
                                        223)
        tumor_image = cv2.imread(os.path.join(patient_path, 'tumor_mask.tif'))
        for i in np.unique(label):
            mask = np.where(label == i, 1, 0)
            array =  mask * tumor_image
            num_feature = pd.DataFrame([array[:,:,0].sum(),array[:,:,1].sum(),array[:,:,2].sum(),array[:,:,3].sum()])
            num_feature.columns = ['The_percentage_of_TCs','The_percentage_of_TILs','The_percentage_of_Stromal','The_percentage_of_Others']
            feature = Pathomic_Feature(array, os.path.join(patient_path, 'B.tif'), os.path.join(patient_path, 'G.tif'),
                                       os.path.join(patient_path, 'R.tif'), mask)
            df = pd.concat([df, num_feature, feature], axis=0)


        df.to_excel('E:\\wsi\\test\\node\\'+path+'.xlsx',index=False)
        edge = get_edge(label)
        edge.to_excel('E:\\wsi\\test\\edge\\'+path+'.xlsx',index=False)



if __name__ == '__main__':
    main()


