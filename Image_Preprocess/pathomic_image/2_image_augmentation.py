import cv2
import numpy as np
import os

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

def main():
    wsi_path = 'E:\\wsi\\test\\'
    for path in os.listdir(wsi_path):
        m = 0
        patient_path = os.path.join(wsi_path, path)
        os.mkdir(patient_path + '_35')
        img1 = cv2.flip(cv2.imread(os.path.join(patient_path, 'mask.tif')), 1)
        img2 = cv2.flip(cv2.imread(os.path.join(patient_path, 'tumor_mask.tif')), 1)
        img3 = cv2.flip(cv2.imread(os.path.join(patient_path, 'immune_mask.tif')), 1)
        img4 = cv2.flip(cv2.imread(os.path.join(patient_path, 'other_mask.tif')), 1)
        img5 = cv2.flip(cv2.imread(os.path.join(patient_path, 'fibroblast_mask.tif')), 1)
        img6 = cv2.flip(cv2.imread(os.path.join(patient_path, 'cell_image.tif')), 1)
        img7 = cv2.flip(cv2.imread(os.path.join(patient_path, 'B.tif')), 1)
        img8 = cv2.flip(cv2.imread(os.path.join(patient_path, 'G.tif')), 1)
        img9 = cv2.flip(cv2.imread(os.path.join(patient_path, 'R.tif')), 1)
        cv2.imwrite(os.path.join(patient_path + '_35', 'mask.tif'), img1)
        cv2.imwrite(os.path.join(patient_path + '_35', 'tumor_mask.tif'), img2)
        cv2.imwrite(os.path.join(patient_path + '_35', 'immune_mask.tif'), img3)
        cv2.imwrite(os.path.join(patient_path + '_35', 'other_mask.tif'), img4)
        cv2.imwrite(os.path.join(patient_path + '_35', 'fibroblast_mask.tif'), img5)
        cv2.imwrite(os.path.join(patient_path + '_35', 'cell_image.tif'), img6)
        cv2.imwrite(os.path.join(patient_path + '_35', 'B.tif'), img7)
        cv2.imwrite(os.path.join(patient_path + '_35', 'G.tif'), img8)
        cv2.imwrite(os.path.join(patient_path + '_35', 'R.tif'), img9)

        for i in range(1, 18):
            m += 1
            os.mkdir(patient_path + '_' + str(m))
            angle = m * 20
            img1 = rotate_bound(cv2.imread(os.path.join(patient_path, 'mask.tif')), angle)
            img2 = rotate_bound(cv2.imread(os.path.join(patient_path, 'tumor_mask.tif')), angle)
            img3 = rotate_bound(cv2.imread(os.path.join(patient_path, 'immune_mask.tif')), angle)
            img4 = rotate_bound(cv2.imread(os.path.join(patient_path, 'other_mask.tif')), angle)
            img5 = rotate_bound(cv2.imread(os.path.join(patient_path, 'fibroblast_mask.tif')), angle)
            img6 = rotate_bound(cv2.imread(os.path.join(patient_path, 'cell_image.tif')), angle)
            img7 = rotate_bound(cv2.imread(os.path.join(patient_path, 'B.tif')), angle)
            img8 = rotate_bound(cv2.imread(os.path.join(patient_path, 'G.tif')), angle)
            img9 = rotate_bound(cv2.imread(os.path.join(patient_path, 'R.tif')), angle)
            cv2.imwrite(os.path.join(patient_path + '_' + str(m), 'mask.tif'), img1)
            cv2.imwrite(os.path.join(patient_path + '_' + str(m), 'tumor_mask.tif'), img2)
            cv2.imwrite(os.path.join(patient_path + '_' + str(m), 'immune_mask.tif'), img3)
            cv2.imwrite(os.path.join(patient_path + '_' + str(m), 'other_mask.tif'), img4)
            cv2.imwrite(os.path.join(patient_path + '_' + str(m), 'fibroblast_mask.tif'), img5)
            cv2.imwrite(os.path.join(patient_path + '_' + str(m), 'cell_image.tif'), img6)
            cv2.imwrite(os.path.join(patient_path + '_' + str(m), 'B.tif'), img7)
            cv2.imwrite(os.path.join(patient_path + '_' + str(m), 'G.tif'), img8)
            cv2.imwrite(os.path.join(patient_path + '_' + str(m), 'R.tif'), img9)

            n = m + 17
            os.mkdir(patient_path + '_' + str(n))
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
            img3 = cv2.flip(img3, 1)
            img4 = cv2.flip(img4, 1)
            img5 = cv2.flip(img5, 1)
            img6 = cv2.flip(img6, 1)
            img7 = cv2.flip(img7, 1)
            img8 = cv2.flip(img8, 1)
            img9 = cv2.flip(img9, 1)

            cv2.imwrite(os.path.join(patient_path + '_' + str(n), 'mask.tif'), img1)
            cv2.imwrite(os.path.join(patient_path + '_' + str(n), 'tumor_mask.tif'), img2)
            cv2.imwrite(os.path.join(patient_path + '_' + str(n), 'immune_mask.tif'), img3)
            cv2.imwrite(os.path.join(patient_path + '_' + str(n), 'other_mask.tif'), img4)
            cv2.imwrite(os.path.join(patient_path + '_' + str(n), 'fibroblast_mask.tif'), img5)
            cv2.imwrite(os.path.join(patient_path + '_' + str(n), 'cell_image.tif'), img6)
            cv2.imwrite(os.path.join(patient_path + '_' + str(n), 'B.tif'), img7)
            cv2.imwrite(os.path.join(patient_path + '_' + str(n), 'G.tif'), img8)
            cv2.imwrite(os.path.join(patient_path + '_' + str(n), 'R.tif'), img9)

if __name__ == '__main__':
    main()



