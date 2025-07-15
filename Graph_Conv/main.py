from models import Unimodal_Model,Multimodal_Model


if __name__ == '__main__':
    epochs = 500
    for epoch in range(epochs):
        Radiomic_Model = Unimodal_Model('E:\\train_label.xlsx','E:\\dce\\root\\train','E:\\dce\\train\\node',
                                        'E:\\dce\\train\\edge','E:\\dce\\model', 135, 32, 24, epochs, epoch)
        Pathomic_Model = Unimodal_Model('E:\\train_label.xlsx','E:\\wsi\\root\\train','E:\\wsi\\train\\node',
                                        'E:\\wsi\\train\\edge','E:\\wsi\\model', 184, 36, 24, epochs, epoch)
        Con_Model = Multimodal_Model('E:\\train_label.xlsx','E:\\dce\\root','E:\\dce\\train\\node',
                                     'E:\\wsi\\dce\\edge','E:\\wsi\\root','E:\\wsi\\train\\node',
                                     'E:\\wsi\\train\\edge','E:\\model', 135, 32, 24,
                                     184, 36, 24, epochs, epoch)
