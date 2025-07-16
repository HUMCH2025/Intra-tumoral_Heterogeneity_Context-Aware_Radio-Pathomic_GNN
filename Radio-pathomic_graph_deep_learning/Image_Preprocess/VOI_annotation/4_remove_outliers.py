import numpy as np
import os
import nibabel as nib

def traversalDir_FirstDir(path):
    list1 = []
    if (os.path.exists(path)):
        files = os.listdir(path)
        for file in files:
            m = os.path.join(path,file)
            list1.append(m)
    return(list1)

def make_image_mean_3std(ROIpath,imagepath):
    ROI = nib.load(ROIpath)
    ROI_data = ROI.get_fdata()
    x,y,z=np.nonzero(ROI_data)
    image =['curve_shape_index.nii.gz','Enhancement at first postcontrast time point.nii.gz',
            'maximum_enhancement.nii.gz','SER.nii.gz','time_to_peak.nii.gz','uptake_rate.nii.gz',
            'washout_rate.nii.gz']
    new_array = np.zeros([len(x),len(image)])
    for i in range(len(image)):
        data_image = nib.load(imagepath+"\\"+image[i])
        data_affine = data_image.affine.copy()
        data_hdr = data_image.header.copy()
        data = data_image.get_fdata()
        for j in range(len(x)):
            new_array[j,i]=data[x[j],y[j],z[j]]
        a_list = new_array[:,i]
        mean1 = np.mean(a_list)
        std1 = np.std(a_list)
        for k in range(len(x)):
            if new_array[k,i] < (mean1-3*std1):
                new_array[k,i] = mean1-3*std1
            if new_array[k,i] > (mean1+3*std1):
                new_array[k,i] = mean1+3*std1
            data[x[k],y[k],z[k]] = new_array[k,i]
        if ROIpath.split("\\")[-1] =='DCE-label.nii.gz':
            nib.Nifti1Image(data,data_affine,data_hdr).to_filename(imagepath+'\\'+ image[i])

def make_imgs(path0):
    patientlist = traversalDir_FirstDir(path0)
    for patient in patientlist:
        print(patient)
        ROIpath1=patient+'\\'+ 'DCE-label.nii.gz'
        make_image_mean_3std(ROIpath1,patient)

def main():
    path0 = 'E:\\dce_resize\\test'
    make_imgs(path0)

if __name__ == '__main__':
    main()
