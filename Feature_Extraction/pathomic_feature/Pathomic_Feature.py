import cv2
import math
import networkx as nx
import SimpleITK as sitk
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from radiomics import shape2D,firstorder,glcm
from shapely.geometry import Polygon
from scipy.spatial import Voronoi, Delaunay
from networkx.algorithms.community import greedy_modularity_communities

def none_list(List):
    if len(List) == 0:
        N = [0]
        return N
    else:
        return List

def compute_perimeter(vor, point_indices):
    perimeter = 0
    for i, p in enumerate(point_indices):
        x, y = vor.vertices[p]
        # calculate perimeter
        if i == len(point_indices) - 1:
            next_p = point_indices[0]
        else:
            next_p = point_indices[i + 1]
        next_x, next_y = vor.vertices[next_p]
        perimeter += np.sqrt((x - next_x) ** 2 + (y - next_y) ** 2)
    return perimeter

def compute_area(vor, point_indices):
    points = []
    for i, p in enumerate(point_indices):
        if i == len(point_indices) - 1:
            points.append(vor.vertices[p])
            points.append(vor.vertices[point_indices[0]])
        else:
            points.append(vor.vertices[p])
    area = Polygon(np.array(points)).area
    return area

def compute_chord_lengths(vor, point_indices):
    chord_length = []
    for i, p in enumerate(point_indices):
        x, y = vor.vertices[p]
        for j in range(i + 2, len(point_indices)):
            if j - i == len(point_indices) - 1:
                pass
            else:
                next_p = point_indices[j]
                next_x, next_y = vor.vertices[next_p]
                length = np.sqrt((x - next_x) ** 2 + (y - next_y) ** 2)
                chord_length.append(length)
    return chord_length

def find_center_and_radius(vor, region):
    points = np.array([vor.vertices[p] for p in region])
    center = np.mean(points, axis=0)
    radius = np.min([np.linalg.norm(p - center) for p in points])
    return center, radius

def calculate_perimeter(x1, x2, x3):
    return np.sqrt((x2[0] - x1[0]) ** 2 + (x2[1] - x1[1]) ** 2) + np.sqrt(
        (x3[0] - x2[0]) ** 2 + (x3[1] - x2[1]) ** 2) + np.sqrt((x1[0] - x3[0]) ** 2 + (x1[1] - x3[1]) ** 2)

def calculate_area(x1, x2, x3):
    points = np.array([x1, x2, x3, x1])
    area = Polygon(points).area
    return area

def calculate_cell_weights_and_num(points):
    num_list50 = []
    num_list70 = []
    weights5 = []
    weights7 = []
    for i in range(len(points)):
        weight_list = []
        for k in range(len(points)):
            if i == k:
                pass
            else:
                weight_list.append(np.linalg.norm(np.array(points[i]) - np.array(points[k])))
        weight_list.sort()
        t50, t70 = 0, 0
        weights5.append(sum(weight_list[0:5]))
        weights7.append(sum(weight_list[0:7]))
        for weight in weight_list:
            if weight <= 70:
                t70 = t70 + 1
            if weight <= 50:
                t50 = t50 + 1
        num_list50.append(t50)
        num_list70.append(t70)
    return num_list50, num_list70, weights5, weights7

def make_length(point1, point2):
    length = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return length

def make_matrix(list1, r, a):
    matrix = np.zeros((len(list1), len(list1)))
    nx_point = []
    for i in range(len(list1)):
        for j in range(i + 1, len(list1)):
            try:
                length = make_length(list1[i], list1[j])
                if r < length ** (-a):
                    matrix[i, j] = 1
                    x = (i, j)
                    nx_point.append(x)
                else:
                    pass
            except Exception as e:
                print(i, j, e)
    matrix_t = matrix.T
    matrix = matrix + matrix_t
    return matrix, nx_point

def make_oc_direct_matrix(cell_infor, a_matrix, notes_list):
    oc_matrix_list = []
    for notes in notes_list:
        notes1 = sorted(notes)
        oc_matrix = np.zeros((18, 18))
        for i in range(len(notes1)):
            for k in range(i + 1, len(notes1)):
                if a_matrix[notes1[i], notes[k]] == 1:
                    a = int(cell_infor.iloc[notes1[i], -1])
                    b = int(cell_infor.iloc[notes1[k], -1])
                    oc_matrix[a - 1, b - 1] = oc_matrix[a - 1, b - 1] + 1
                else:
                    pass
        oc_matrix1 = oc_matrix + oc_matrix.T
        oc_matrix_list.append(oc_matrix1)
    return oc_matrix_list


def CORE_feature(data_array):
    contrast_inverse_moment = 0
    N = data_array.shape[0]
    Pn = dict()
    num_n = dict()
    Qk = dict()
    num_k = dict()
    HXY1 = 0
    HXY2 = 0
    for a in range(N):
        for b in range(N):
            for n in range(N):
                contrast_inverse_moment = contrast_inverse_moment + (1 / (1 + (a - b) ** 2)) * data_array[a, b]
                if abs(a - b) == n:
                    num_n[n] = num_n.get(n, 0) + 1
                    Pn[n] = Pn.get(n, 0) + data_array[a, b]
            for k in range(2, N + N + 1):
                if abs(a + 1 + b + 1) == k:
                    num_k[k] = num_k.get(k, 0) + 1
                    Qk[k] = Qk.get(k, 0) + data_array[a, b]
                else:
                    pass
    contrast_energy = sum(n ** 2 * Pn[n] for n in Pn)
    contrast_average = sum(n ** 2 * num_n[n] * Pn[n] for n in Pn)
    contrast_variance = sum(num_n[n] * (n - contrast_average) ** 2 * Pn[n] for n in Pn)
    intensity_average = sum(k * Qk[k] for k in Qk)
    contrast_entropy = 0
    for n1 in Pn:
        if Pn[n1] > 0:
            contrast_entropy = contrast_entropy + (-Pn[n1] * math.log2(Pn[n1]))
        else:
            pass
    intensity_entropy = 0
    for k1 in Qk:
        if Qk[k1] > 0:
            intensity_entropy = intensity_entropy + (-Qk[k1] * math.log2(Qk[k1]))
        else:
            pass
    intensity_variance = sum((k - intensity_entropy) ** 2 * Qk[k] for k in Qk)
    flattened = data_array.flatten()
    probabilities = flattened[flattened > 0]
    entropy = np.sum(-probabilities * np.log2(probabilities))
    energy = np.sum(probabilities ** 2)

    px = np.sum(data_array, axis=1)
    py = np.sum(data_array, axis=0)
    mean_x = np.mean(data_array, axis=1)
    mean_y = np.mean(data_array, axis=0)
    x_cha = px - mean_x
    y_cha = py - mean_y
    std_x = np.std(data_array, axis=1)
    std_y = np.std(data_array, axis=0)
    correlation = 0
    for i in range(mean_x.shape[0]):
        for j in range(mean_y.shape[0]):
            t = px[i] * py[j]
            if t == 0:
                pass
            else:
                HXY1 = HXY1 + (-data_array[i, j] * math.log2(t))
                HXY2 = HXY2 + (-t * math.log2(t))

            if (std_x[i] * std_y[j]) == 0:
                pass
            else:
                correlation = correlation + (x_cha[i] * y_cha[j] * data_array[i, j]) / (std_x[i] * std_y[j])
    return [contrast_energy, contrast_inverse_moment, contrast_average, contrast_variance, contrast_entropy,
            intensity_average, intensity_variance, intensity_entropy, entropy, energy, correlation]

class Pathomic_Feature(object):
    def __init__(self,array,B_path,G_path,R_path,mask):
        self.settings = \
            {'binCount': 32,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None,
            'normalize': True,
            'normalizeScale': 255,
            'label': 1
            }
        self.mask = mask
        self.B_img = sitk.ReadImage(B_path)
        self.G_img = sitk.ReadImage(G_path)
        self.R_img = sitk.ReadImage(R_path)
        _, self.location_list, _ = list(cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE))
        self.R = 0.3
        self.A = 0.1

    def get_features_vd(self):
        areas = []
        perimeters = []
        chord_lengths = []
        cell_point = []
        for l in self.mask:
            M = cv2.moments(np.array(l))
            X = int(M["m10"]/(M["m00"]-1e-6))
            Y = int(M["m01"]/(M["m00"]-1e-6))
            if self.array[X,Y] == 1:
                cell_point.append((X,Y))
        if len(cell_point) < 4:
            return [0,0,0,0,0,0,0,0,0]
        vor = Voronoi(cell_point)
        for region in vor.regions:
            if -1 not in region:
                if len(region) == 0:
                    pass
                else:
                    p = compute_perimeter(vor, region)
                    a = compute_area(vor,region)
                    chord_length = compute_chord_lengths(vor,region)
                    perimeters.append(p)
                    areas.append(a)
                    if len(chord_length) == 0 :
                        chord_lengths.append(0)
                    else:
                        chord_lengths.append(max(chord_length))
        areas = np.array(none_list(areas))
        perimeters = np.array(none_list(perimeters))
        chord_lengths = np.array(none_list(chord_lengths))
        feature = pd.DataFrame([areas.mean(),areas.std(),perimeters.mean(),perimeters.std(),chord_lengths.mean(),chord_lengths.std()])
        feature.columns = ['vd_area_mean','vd_area_std','vd_perimeter_mean','vd_perimeter_std','vd_max_distance_mean','vd_max_distance_std']
        return feature

    def get_features_dt(self):
        areas = []
        perimeters = []
        cell_point = []
        for l in self.mask:
            M = cv2.moments(np.array(l))
            X = int(M["m10"] / (M["m00"] - 1e-6))
            Y = int(M["m01"] / (M["m00"] - 1e-6))
            if self.array[X, Y] == 1:
                cell_point.append((X, Y))
        if len(cell_point) < 3:
            perimeters.append(0)
            areas.append(0)
        else:
            D = Delaunay(cell_point)
            perimeters = []
            areas = []
            for num, simplex in enumerate(D.simplices):
                x1 = cell_point[simplex[0]]
                x2 = cell_point[simplex[1]]
                x3 = cell_point[simplex[2]]
                perimeters.append(calculate_perimeter(x1, x2, x3))
                areas.append(calculate_area(x1, x2, x3))
        area = np.array(none_list(areas))
        perimeter = np.array(none_list(perimeters))
        feature = pd.DataFrame([area.mean(), area.std(), perimeter.mean(), perimeter.std()])
        feature.columns = ['dt_area_mean','dt_area_std','dt_perimeter_mean','dt_perimeter_std']
        return feature

    def get_features_mst(self):
        edge_length = []
        cell_point = []
        for l in self.mask:
            M = cv2.moments(np.array(l))
            X = int(M["m10"] / (M["m00"] - 1e-6))
            Y = int(M["m01"] / (M["m00"] - 1e-6))
            if self.array[X, Y] == 1:
                cell_point.append((X, Y))
        if len(cell_point) < 3:
            edge_length.append(0)
        else:
            tri = Delaunay(cell_point)
            G = nx.Graph()
            for i in range(len(cell_point)):
                G.add_node(i)
            for a, b, c in tri.simplices:
                G.add_edge(a, b, weight=np.linalg.norm(np.array(cell_point[a]) - np.array(cell_point[b])))
                G.add_edge(a, c, weight=np.linalg.norm(np.array(cell_point[a]) - np.array(cell_point[c])))
                G.add_edge(b, c, weight=np.linalg.norm(np.array(cell_point[b]) - np.array(cell_point[c])))
            mst = nx.minimum_spanning_tree(G)
            points_weight = sorted(mst.edges(data=True))
            for k in range(len(points_weight)):
                edge_length.append(points_weight[k][2]['weight'])
        edge_length = np.array(none_list(edge_length))
        feature = pd.DataFrame([edge_length.mean(), edge_length.std()])
        feature.columns = ['mst_edge_length_mean','mst_edge_length_std']
        return feature

    def get_features_nd(self):
        cell_point = []
        for l in self.mask:
            M = cv2.moments(np.array(l))
            X = int(M["m10"] / (M["m00"] - 1e-6))
            Y = int(M["m01"] / (M["m00"] - 1e-6))
            if self.array[X, Y] == 1:
                cell_point.append((X, Y))
        num_list50, num_list70, weights5, weights7 = calculate_cell_weights_and_num(cell_point)
        weights5 = np.array(none_list(weights5))
        weights7 = np.array(none_list(weights7))
        num_list50 = np.array(none_list(num_list50))
        num_list70 = np.array(none_list(num_list70))
        feature = pd.DataFrame([weights5.mean(), weights5.std(),
                                weights7.mean(), weights7.std(),
                                num_list50.mean(), num_list50.std(),
                                num_list70.mean(), num_list70.std()])
        feature.columns = ['aver_distance_to_5_nn_mean','aver_distance_to_5_nn_std',
                           'aver_distance_to_7_nn_mean','aver_distance_to_7_nn_std',
                           'number_of_nn_in_50_pr_mean','number_of_nn_in_50_pr_std',
                           'number_of_nn_in_70_pr_mean','number_of_nn_in_70_pr_std']
        return feature

    def get_features_oe(self):
        contrast_energy = []
        contrast_inverse_moment = []
        contrast_average = []
        contrast_variance = []
        contrast_entropy = []
        intensity_average = []
        intensity_variance = []
        intensity_entropy = []
        entropy = []
        energy = []
        correlation = []
        information_measure_1 = []
        information_measure_2 = []
        cell_point = []
        cell_information = []
        for l in self.mask:
            M = cv2.moments(np.array(l))
            X = int(M["m10"] / (M["m00"] - 1e-6))
            Y = int(M["m01"] / (M["m00"] - 1e-6))
            if self.array[X, Y] == 1:
                cell_point.append((X, Y))
                pca = PCA(n_components=1)
                principal_components = pca.fit_transform(l)
                z = principal_components[0]
                theta = 180 / np.pi * np.arctan2(z[1], z[0])
                if theta < 0:
                    theta = theta + 180
                cell_information.append((X, Y, theta / 10))
        a_maxtrix, nx_points = make_matrix(cell_point, self.R, self.A)
        G = nx.Graph()
        G.add_edges_from(nx_points)
        if len(nx_points) == 0:
            contrast_energy.append(0)
            contrast_inverse_moment.append(0)
            contrast_average.append(0)
            contrast_variance.append(0)
            contrast_entropy.append(0)
            intensity_average.append(0)
            intensity_variance.append(0)
            intensity_entropy.append(0)
            entropy.append(0)
            energy.append(0)
            correlation.append(0)
            information_measure_1.append(0)
            information_measure_2.append(0)
        else:
            communities = list(greedy_modularity_communities(G))
            netxnotes_list = []
            for l in communities:
                if len(list(l)) >= 18:
                    netxnotes_list.append(list(l))
            cell_information = pd.DataFrame(cell_information)
            oc_matrix_list = make_oc_direct_matrix(cell_information, a_maxtrix, netxnotes_list)
            for matrix in oc_matrix_list:
                feature = CORE_feature(matrix / np.sum(matrix))
                contrast_energy.append(feature[0])
                contrast_inverse_moment.append(feature[1])
                contrast_average.append(feature[2])
                contrast_variance.append(feature[3])
                contrast_entropy.append(feature[4])
                intensity_average.append(feature[5])
                intensity_variance.append(feature[6])
                intensity_entropy.append(feature[7])
                entropy.append(feature[8])
                energy.append(feature[9])
                correlation.append(feature[10])
        contrast_energy = np.array(none_list(contrast_energy))
        contrast_inverse_moment = np.array(none_list(contrast_inverse_moment))
        contrast_average = np.array(none_list(contrast_average))
        contrast_variance = np.array(none_list(contrast_variance))
        contrast_entropy = np.array(none_list(contrast_entropy))
        intensity_average = np.array(none_list(intensity_average))
        intensity_variance = np.array(none_list(intensity_variance))
        intensity_entropy = np.array(none_list(intensity_entropy))
        entropy = np.array(none_list(entropy))
        energy = np.array(none_list(energy))
        correlation = np.array(none_list(correlation))
        feature = pd.DataFrame([contrast_energy[0],contrast_inverse_moment[0],contrast_average[0],contrast_variance[0],
                                contrast_entropy[0],intensity_average[0],intensity_variance[0],intensity_entropy[0],
                                entropy[0],energy[0],correlation[0]])
        feature.columns = ['contrast_energy','contrast_inverse_moment','contrast_average','contrast_variance',
                           'contrast_entropy','intensity_average','intensity_variance','intensity_entropy',
                           'entropy','energy','correlation']
        return feature

    def get_features_2D(self):
        mask = sitk.GetImageFromArray(self.mask)
        shape_feature = shape2D.RadiomicsShape2D(self.R_img, mask, **self.settings)
        shape_feature.enableAllFeatures()
        shapeVector = shape_feature.execute()
        df_shape = pd.DataFrame.from_dict(shapeVector.values()).T
        df_shape.columns = ['shape_' + n for n in list(shapeVector.keys())]

        # firstorder_Variance
        mask = sitk.GetImageFromArray(self.mask)
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(self.R_img, mask, **self.settings)
        firstorderVector = firstOrderFeatures.execute()
        df_firstorder1 = pd.DataFrame.from_dict(firstorderVector.values()).T
        df_firstorder1.columns = ['firstorder_r_' + n for n in list(firstorderVector.keys())]

        mask = sitk.GetImageFromArray(self.mask)
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(self.G_img, mask, **self.settings)
        firstorderVector = firstOrderFeatures.execute()
        df_firstorder2 = pd.DataFrame.from_dict(firstorderVector.values()).T
        df_firstorder2.columns = ['firstorder_g_' + n for n in list(firstorderVector.keys())]

        mask = sitk.GetImageFromArray(self.mask)
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(self.B_img, mask, **self.settings)
        firstorderVector = firstOrderFeatures.execute()
        df_firstorder3 = pd.DataFrame.from_dict(firstorderVector.values()).T
        df_firstorder3.columns = ['firstorder_b_' + n for n in list(firstorderVector.keys())]

        # glcm
        glcmFeatures = glcm.RadiomicsGLCM(self.R_img, mask, **self.settings)
        glcmFeatures.enableAllFeatures()
        glcmVector = glcmFeatures.execute()
        df_glcm1 = pd.DataFrame.from_dict(glcmVector.values()).T
        df_glcm1.columns = ['glcm_r_' + n for n in list(glcmVector.keys())]

        glcmFeatures = glcm.RadiomicsGLCM(self.G_img, mask, **self.settings)
        glcmFeatures.enableAllFeatures()
        glcmVector = glcmFeatures.execute()
        df_glcm2 = pd.DataFrame.from_dict(glcmVector.values()).T
        df_glcm2.columns = ['glcm_g_' + n for n in list(glcmVector.keys())]

        glcmFeatures = glcm.RadiomicsGLCM(self.B_img, mask, **self.settings)
        glcmFeatures.enableAllFeatures()
        glcmVector = glcmFeatures.execute()
        df_glcm3 = pd.DataFrame.from_dict(glcmVector.values()).T
        df_glcm3.columns = ['glcm_b_' + n for n in list(glcmVector.keys())]

        df_feature = pd.concat([df_shape, df_firstorder1, df_firstorder2, df_firstorder3, df_glcm1, df_glcm2, df_glcm3], axis=1)
        return df_feature

    def get_features(self):
        vd_feature = self.get_features_vd()
        dt_feature = self.get_features_dt()
        mst_feature = self.get_features_mst()
        oe_feature = self.get_features_oe()
        add_feature = self.get_features_2D()
        feature = pd.concat([vd_feature, dt_feature, mst_feature, oe_feature, add_feature], axis=1)
        return feature


