# Context-AwareRadio-Pathomic Graph
####
$~~~~$ Code of our paper __"Context-Aware Radio-Pathomic Graph Deep Learning of Intra-tumoral Heterogeneity for Predicting Breast Cancer Neoadjuvant Therapy Response"__
####
<img src=".\\pic\\Visual_abstract.png"/>

## Procedures

### Radio-pathomic graph deep learning

$~~~~$ You can find all the code in this section in the file ".\\Radio-pathomic_graph_deep_learning"

#### __VOI annotation/ROI annatation__

$~~~~$ __VOI annotation:__ ".\\Radio-pathomic_graph_deep_learning\\Image_Preprocess\\VOI_annotation"

$~~~~$ __ROI annotation:__ ".\\Radio-pathomic_graph_deep_learning\\Image_Preprocess\\ROI_annotation"

#### __Subregion segmentation and Subregion feature extraction__

$~~~~$ __Radiomic:__ ".\\Radio-pathomic_graph_deep_learning\\Feature_Extraction\\Radiomic_feature"

$~~~~$ __Pathomic:__ ".\\Radio-pathomic_graph_deep_learning\\Feature_Extraction\\Pathomic_feature"

#### __Attention-based graph learning__

$~~~~$ ".\\Radio-pathomic_graph_deep_learning\\Attention-based_graph_learning"

### Interpretations

$~~~~$ You can find all the code in this section in the file ".\\Interpretations"


## Requirements

The below operations implemented in Python v.3.9.12
- torch == 1.13.1+cu117
- torchvision == 0.14.1+cu117
- torchaudio == 0.13.1
- torch_geometric == 2.6.1
- numpy == 1.24.4
- pandas == 2.0.3
- opencv-python == 4.10.0.84
- pyradiomics == 3.1.0
- networkx == 3.1
- SimpleITK == 2.3.1
- nibabel == 5.2.1
- scikit-learn == 1.0.2
- shapely == 2.0.3
- scipy == 1.8.1

To install requirements, run `pip install -r requirements.txt`
