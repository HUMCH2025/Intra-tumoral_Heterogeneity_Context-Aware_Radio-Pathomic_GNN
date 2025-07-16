import staintools
import cv2
import os

def main():
    img = cv2.imread('E:\\wsi\\standard.tif')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    normalizer = staintools.StainNormalizer(method='vahadane')
    normalizer.fit(img)

    for path in os.listdir('E:\\wsi\\train\\'):
        image_path = os.path.join(os.path.join('E:\\wsi\\test\\', path), 'ROI.tif')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = normalizer.transform(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        B, G, R = cv2.split(image)
        cv2.imwrite(os.path.join(os.path.join('E:\\wsi\\test\\', path),'norm_ROI.tif'), image)
        cv2.imwrite(os.path.join(os.path.join('E:\\wsi\\test\\', path), 'norm_B.tif'), B)
        cv2.imwrite(os.path.join(os.path.join('E:\\wsi\\test\\', path), 'norm_G.tif'), G)
        cv2.imwrite(os.path.join(os.path.join('E:\\wsi\\test\\', path), 'norm_R.tif'), R)

if __name__ == '__main__':
    main()