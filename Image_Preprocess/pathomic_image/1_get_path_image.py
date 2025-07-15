import cv2
import json
import numpy as np
import os

def integerization(cell_list):
    x_list = []
    y_list = []
    List = []
    for cell in cell_list:
        x_list.append(round(cell[0]))
        y_list.append(round(cell[1]))
    for i in range(len(cell_list)):
        List.append([x_list[i], y_list[i]])
    return np.array(List)
def normalization(cell_list,x,y):
    x_list = []
    y_list = []
    List = []
    for cell in cell_list:
        x_list.append(cell[0]-x)
        y_list.append(cell[1]-y)
    for i in range(len(cell_list)):
        List.append([x_list[i], y_list[i]])
    return np.array(List)
def divide_up(cell_list):
    x_list = []
    y_list = []
    for cell in cell_list:
        x_list.append(cell[0])
        y_list.append(cell[1])
    return np.array(x_list),np.array(y_list)
def tranpose(cell_list):
    x_list = []
    y_list = []
    List = []
    for cell in cell_list:
        x_list.append(cell[0])
        y_list.append(cell[1])
    for i in range(len(cell_list)):
        List.append([y_list[i], x_list[i]])
    return np.array(List)
def get_pos(Max,Min,n):
    if (Max - Min) // n == 0:
        pos = (Max - Min) // n
    else:
        pos = (Max - Min) // n + 1
    return pos


def get_cell_mask(roi_path, path):
    cell_list = []
    cell = json.load(open(roi_path))['features']
    for i in range(len(cell)):
        if cell[i]['properties']['objectType'] == 'annotation':
            cell_list = cell[i]['geometry']['coordinates'][0]
    cell_list = integerization(cell_list)
    roi_x, roi_y = divide_up(cell_list)
    roi = np.zeros((roi_x.max() - roi_x.min(), roi_y.max() - roi_y.min()), np.uint8)
    cv2.fillPoly(roi,[tranpose(cell_list)],255)
    roi = cv2.flip(roi, 1)
    path_mask = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # tumor
    cell_features = json.load(open(path))['features']
    roi = np.zeros((roi_x.max() - roi_x.min(), roi_y.max() - roi_y.min()), np.uint8)
    roi_all = np.zeros((roi_x.max() - roi_x.min(), roi_y.max() - roi_y.min(), 3), np.uint8) + 255
    for features in cell_features:
        if 'nucleusGeometry' in features and features['properties']['classification']['color'] == [255, 0, 0]:  # tumor
            cell_feature = features['nucleusGeometry']['coordinates'][0]
            cell_feature = normalization(cell_feature, roi_x.min(), roi_y.min())
            cell_feature = integerization(cell_feature)
            cv2.drawContours(roi, [tranpose(cell_feature)], -1, 255, 1)
            cv2.drawContours(roi_all, [tranpose(cell_feature)], -1, (0, 0, 255), 3)
    roi = cv2.flip(roi, 1)
    mask1 = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # immune
    cell_features = json.load(open(path))['features']
    roi = np.zeros((roi_x.max() - roi_x.min(), roi_y.max() - roi_y.min()), np.uint8)
    for features in cell_features:
        if 'nucleusGeometry' in features and features['properties']['classification']['color'] == [255, 215, 0]:  # immune
            cell_feature = features['nucleusGeometry']['coordinates'][0]
            cell_feature = normalization(cell_feature, roi_x.min(), roi_y.min())
            cell_feature = integerization(cell_feature)
            cv2.drawContours(roi, [tranpose(cell_feature)], -1, 255, 1)
            cv2.drawContours(roi_all, [tranpose(cell_feature)], -1, (0, 147, 0), 3)
    roi = cv2.flip(roi, 1)
    mask2 = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # other
    cell_features = json.load(open(path))['features']  # 确定细胞信息
    cell_list = integerization(cell_list)
    roi_x, roi_y = divide_up(cell_list)
    roi = np.zeros((roi_x.max() - roi_x.min(), roi_y.max() - roi_y.min()), np.uint8)
    for features in cell_features:
        if 'nucleusGeometry' in features and features['properties']['classification']['color'] == [0, 255, 127]:  # other
            cell_feature = features['nucleusGeometry']['coordinates'][0]
            cell_feature = normalization(cell_feature, roi_x.min(), roi_y.min())
            cell_feature = integerization(cell_feature)
            cv2.drawContours(roi, [tranpose(cell_feature)], -1, 255, 1)
            cv2.drawContours(roi_all, [tranpose(cell_feature)], -1, (65, 150, 217), 3)
    roi = cv2.flip(roi, 1)
    mask3 = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # fibroblast
    cell_features = json.load(open(path))['features']  # 确定细胞信息
    cell_list = integerization(cell_list)
    roi_x, roi_y = divide_up(cell_list)
    roi = np.zeros((roi_x.max() - roi_x.min(), roi_y.max() - roi_y.min()), np.uint8)
    for features in cell_features:
        if 'nucleusGeometry' in features and features['properties']['classification']['color'] == [0, 0, 205]:  # fibroblast
            cell_feature = features['nucleusGeometry']['coordinates'][0]
            cell_feature = normalization(cell_feature, roi_x.min(), roi_y.min())
            cell_feature = integerization(cell_feature)
            cv2.drawContours(roi, [tranpose(cell_feature)], -1, 255, 1)
            cv2.drawContours(roi_all, [tranpose(cell_feature)], -1, (14, 217, 247), 3)
    roi = cv2.flip(roi, 1)
    mask4 = cv2.rotate(roi, cv2.ROTATE_90_COUNTERCLOCKWISE)
    roi_all = cv2.flip(roi_all, 1)
    roi_all = cv2.rotate(roi_all, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return path_mask, mask1, mask2, mask3, mask4, roi_all

def main():
    wsi_path = 'E:\\wsi\\test\\'
    for path in os.listdir(wsi_path):
        patient_path = os.path.join(wsi_path, path)
        image_path = os.path.join(patient_path, 'ROI.json')
        roi_path = os.path.join(patient_path, 'cell.json')
        cell_path = os.path.join(patient_path, 'classify_cell.json')
        image_B, image_G, image_R = cv2.split(image_path)
        path_mask, mask1, mask2, mask3, mask4, roi_all = get_cell_mask(roi_path, cell_path)
        cv2.imwrite(os.path.join(patient_path, 'B.tif'), image_B)
        cv2.imwrite(os.path.join(patient_path, 'G.tif'), image_G)
        cv2.imwrite(os.path.join(patient_path, 'R.tif'), image_R)
        cv2.imwrite(os.path.join(patient_path, 'mask.tif'), path_mask)
        cv2.imwrite(os.path.join(patient_path, 'tumor_mask.tif'), mask1)
        cv2.imwrite(os.path.join(patient_path, 'immune_mask.tif'), mask2)
        cv2.imwrite(os.path.join(patient_path, 'other_mask.tif'), mask3)
        cv2.imwrite(os.path.join(patient_path, 'fibroblast_mask.tif'), mask4)
        cv2.imwrite(os.path.join(patient_path, 'cell_image.tif'), roi_all)


if __name__ == '__main__':
    main()

