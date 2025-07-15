import os
import numpy as np
import pandas as pd
import nibabel as nib
from itertools import combinations
from shutil import copy

def add_gaussian_noise(image):
    sigma = image.mean()
    noise = np.random.normal(0,sigma,image.shape)
    noise_image = image + noise
    return noise_image

def main():
    label = pd.read_excel('E:\\dce\\test_label.xlsx')
    dce_path = 'E:\\dce\\test'
    dce_list = ['curve_shape_index.nii.gz','Enhancement at first postcontrast time point.nii.gz',
                'maximum_enhancement.nii.gz','SER.nii.gz',
                'time_to_peak.nii.gz','uptake_rate.nii.gz',
                'washout_rate.nii.gz']
    comb_list = list(combinations(dce_list,3))
    for path in os.listdir(dce_path):
        n = 0
        patient_path = os.path.join(dce_path, path)
        for change_list in comb_list:
            n += 1
            nochange_list = list(set(dce_list) - set(change_list))
            new_patient_name = path +'_'+ str(n)
            new_patient_path = os.path.join(dce_path, new_patient_name)
            os.mkdir(new_patient_path)
            new_label = pd.DataFrame([path +'_'+ str(n), label[label['name'] == path]['label'].index[0]])
            label = pd.concat([label, new_label], axis=0)
            for new_image in change_list:
                nii_img = nib.load(os.path.join(patient_path, new_image))
                nii_data = nii_img.get_fdata()
                new_nii_data = add_gaussian_noise(nii_data)
                new_nii = nib.Nifti1Image(new_nii_data, nii_img.affine, nii_img.hdr)
                nib.save(new_nii, os.path.join(new_patient_path, new_image))
            for old_image in nochange_list:
                copy(os.path.join(patient_path, old_image), os.path.join(new_patient_path, old_image))
    label.to_excel('D:\\DCE\\test_label.xlsx')


if __name__ == '__main__':
    main()




