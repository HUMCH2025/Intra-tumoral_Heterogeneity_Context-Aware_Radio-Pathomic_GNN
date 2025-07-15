import SimpleITK as sitk
import os

def resampleImage(image, targetSpacing, resamplemethod):
    """
    The specified spacing size for the volume data to be resamphored
    paras：
    image: the image information read by sitk, in this case, the volume data
    targetSpacing: specifies spacing, for example, [1,1,1]
    resampleMethod: the type of interpolation
    return: resampled data
    """
    targetsize = [0, 0, 0]
    #Read the size and spacing information of the raw data
    ori_size = image.GetSize()
    ori_spacing = image.GetSpacing()
    transform = sitk.Transform()
    transform.SetIdentity()
    #Use the physical size/voxel size to calculate the size after changing spacing
    targetsize[0] = round(ori_size[0] * ori_spacing[0] / targetSpacing[0])
    targetsize[1] = round(ori_size[1] * ori_spacing[1] / targetSpacing[1])
    targetsize[2] = round(ori_size[2] * ori_spacing[2] / targetSpacing[2])
    #Set some parameters for resampling
    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(transform)
    resampler.SetSize(targetsize)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputSpacing(targetSpacing)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(resamplemethod)
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt16)
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)
    newImage = resampler.Execute(image)
    return newImage

def main():
    #The nii files from seven pharmacokinetic parameters
    img_list = ['curve_shape_index.nii.gz', 'maximum_enhancement.nii.gz',
                'Enhancement at first postcontrast time point.nii.gz',
                'SER.nii.gz', 'time_to_peak.nii.gz', 'uptake_rate.nii.gz',
                'washout_rate.nii.gz', 'DCE-label.nii.gz']
    #The original address where the files were stored
    path = 'E:\\dce\\test'
    #the resampling address where the file were stored
    new_path = 'E:\\dce_resize\\test'
    for nii_path in os.listdir(path):
        nii = path + "\\" + nii_path
        if not os.path.exists(new_path + "\\" + nii_path):
            os.mkdir(new_path + "\\" + nii_path)
        for image in img_list:
            img = nii + "\\" + image
            img = sitk.ReadImage(img)
            #'DCE-label.nii.gz' is cancer segmentation result from radiologists using 3D Slicer
            if image == 'DCE-label.nii.gz':
                img = resampleImage(img, [1, 1, 1], sitk.sitkUInt8)
            else:
                img = resampleImage(img, [1, 1, 1], sitk.sitkFloat32)
            img_array = sitk.GetArrayFromImage(img)
            saveImg = sitk.GetImageFromArray(img_array)
            saveImg.SetOrigin(img.GetOrigin())
            saveImg.SetDirection(img.GetDirection())
            saveImg.SetSpacing(img.GetSpacing())
            sitk.WriteImage(saveImg, new_path + "\\" + nii_path + "\\" + image)
            print(new_path + "\\" + nii_path + "\\" + image + " is done!")


if __name__ == '__main__':
    main()