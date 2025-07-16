import os
import nibabel as nib
import numpy as np


def kinetic_curves(input_path,output_path):
    for path in os.listdir(input_path):
        patient_path = os.path.join(input_path, path)
        final_path = os.path.join(output_path, path)
        S0path = nib.load(os.path.join(patient_path, '0.nii.gz'))
        S0 = S0path.get_fdata()
        affine = S0path.affine.copy()
        hdr = S0path.header.copy()
        S1 = nib.load(os.path.join(patient_path, '1.nii.gz')).get_fdata()
        S2 = nib.load(os.path.join(patient_path, '2.nii.gz')).get_fdata()
        S3 = nib.load(os.path.join(patient_path, '3.nii.gz')).get_fdata()
        S4 = nib.load(os.path.join(patient_path, '4.nii.gz')).get_fdata()
        #S5 = nib.load(patientpath +'\\'+index_data+ '\\5.nii.gz').get_fdata()
        data = np.stack((S0,S1,S2,S3,S4),axis=0)
        new_data = data.reshape(5,-1)
        #time=np.array([1e-10,90.6,181.2,270.2,359.3,448.3]).reshape(6,1)
        time=np.array([1e-10,199,268,337,406]).reshape(5,1)
        index=np.argmax(new_data,axis=0)

        time_to_peak=time.reshape(-1,)[index]
        time_to_peak = time_to_peak.reshape(S0.shape[0],S0.shape[1],S0.shape[2])

        curve_shape_index = S4 - S1

        cols_max = np.max(new_data,axis=0)
        maximum_enhancement = (cols_max - new_data[0,])/(new_data[0,]+1e-10)
        maximum_enhancement = maximum_enhancement.reshape(S0.shape[0],S0.shape[1],S0.shape[2])

        uptake_rate=maximum_enhancement / time_to_peak

        mask=(index==4).reshape(1,-1)
        cols_max = cols_max.reshape(1,-1)

        washout_rate = (cols_max-new_data[4,])/((new_data[0,]*(4-index.reshape(1,-1)))+1e-10)
        washout_rate = washout_rate*(1-mask)
        washout_rate = washout_rate.reshape(S0.shape[0],S0.shape[1],S0.shape[2])

        SER = (S1-S0) / (S4-S0+1e-10)
        Enhancement = S2-S1

        nib.save(nib.Nifti1Image(time_to_peak,affine,hdr), os.path.join(final_path, '\\time_to_peak.nii.gz'))
        nib.save(nib.Nifti1Image(curve_shape_index,affine,hdr), os.path.join(final_path, '\\curve_shape_index.nii.gz'))
        nib.save(nib.Nifti1Image(maximum_enhancement,affine,hdr), os.path.join(final_path, '\\maximum_enhancement.nii.gz'))
        nib.save(nib.Nifti1Image(uptake_rate,affine,hdr), os.path.join(final_path, '\\uptake_rate.nii.gz'))
        nib.save(nib.Nifti1Image(washout_rate,affine,hdr), os.path.join(final_path, '\\washout_rate.nii.gz'))
        nib.save(nib.Nifti1Image(SER,affine,hdr), os.path.join(final_path, '\\SER.nii.gz'))
        nib.save(nib.Nifti1Image(Enhancement,affine,hdr), os.path.join(final_path, '\\Enhancement at first postcontrast time point.nii.gz'))

if __name__ == '__main__':
    nii_path = 'E:\\nii\\test'
    dce_path = 'E:\\dce\\test'
    kinetic_curves(nii_path,dce_path)