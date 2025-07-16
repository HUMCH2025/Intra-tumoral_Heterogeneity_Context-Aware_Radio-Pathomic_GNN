import SimpleITK as sitk
from radiomics import shape2D,firstorder,glcm
import pandas as pd

class Radiomics_Feature(object):
    def __init__(self,image_component1,image_component2,image_component3,mask):
        self.settings = \
            {'binCount': 32,
            'interpolator': sitk.sitkBSpline,
            'resampledPixelSpacing': None,
            'normalize': True,
            'normalizeScale': 1,
            'label': 1
            }
        self.image = image_component1
        self.image1 =  image_component1
        self.image2 = image_component2
        self.image3 = image_component3
        self.mask = mask


    def get_features(self):
        shape_feature = shape2D.RadiomicsShape2D(sitk.GetImageFromArray(self.image), sitk.GetImageFromArray(self.mask), **self.settings)
        shape_feature.enableAllFeatures()
        shapeVector = shape_feature.execute()
        df_shape = pd.DataFrame.from_dict(shapeVector.values()).T
        df_shape.columns = ['shape_' + n for n in list(shapeVector.keys())]


        # firstorder_Variance
        firstOrderFeatures = firstorder.RadiomicsFirstOrder(sitk.GetImageFromArray(self.image1), sitk.GetImageFromArray(self.mask), **self.settings)
        firstorderVector = firstOrderFeatures.execute()
        df_firstorder1 = pd.DataFrame.from_dict(firstorderVector.values()).T
        df_firstorder1.columns = ['firstorder1_' + n for n in list(firstorderVector.keys())]

        firstOrderFeatures = firstorder.RadiomicsFirstOrder(sitk.GetImageFromArray(self.image2), sitk.GetImageFromArray(self.mask), **self.settings)
        firstorderVector = firstOrderFeatures.execute()
        df_firstorder2 = pd.DataFrame.from_dict(firstorderVector.values()).T
        df_firstorder2.columns = ['firstorder2_' + n for n in list(firstorderVector.keys())]

        firstOrderFeatures = firstorder.RadiomicsFirstOrder(sitk.GetImageFromArray(self.image3), sitk.GetImageFromArray(self.mask), **self.settings)
        firstorderVector = firstOrderFeatures.execute()
        df_firstorder3 = pd.DataFrame.from_dict(firstorderVector.values()).T
        df_firstorder3.columns = ['firstorder3_' + n for n in list(firstorderVector.keys())]


        # glcm
        glcmFeatures = glcm.RadiomicsGLCM(sitk.GetImageFromArray(self.image1), sitk.GetImageFromArray(self.mask), **self.settings)
        glcmFeatures.enableAllFeatures()
        glcmVector = glcmFeatures.execute()
        df_glcm1 = pd.DataFrame.from_dict(glcmVector.values()).T
        df_glcm1.columns = ['glcm1_' + n for n in list(glcmVector.keys())]

        glcmFeatures = glcm.RadiomicsGLCM(sitk.GetImageFromArray(self.image2), sitk.GetImageFromArray(self.mask), **self.settings)
        glcmFeatures.enableAllFeatures()
        glcmVector = glcmFeatures.execute()
        df_glcm2 = pd.DataFrame.from_dict(glcmVector.values()).T
        df_glcm2.columns = ['glcm2_' + n for n in list(glcmVector.keys())]

        glcmFeatures = glcm.RadiomicsGLCM(sitk.GetImageFromArray(self.image3), sitk.GetImageFromArray(self.mask), **self.settings)
        glcmFeatures.enableAllFeatures()
        glcmVector = glcmFeatures.execute()
        df_glcm3 = pd.DataFrame.from_dict(glcmVector.values()).T
        df_glcm3.columns = ['glcm3_' + n for n in list(glcmVector.keys())]

        df_feature = pd.concat([df_shape, df_firstorder1, df_firstorder2, df_firstorder3, df_glcm1, df_glcm2, df_glcm3], axis=1)
        return df_feature