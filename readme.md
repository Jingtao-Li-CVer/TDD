## One-Step Detection Paradigm for Hyperspectral Anomaly Detection via Spectral Deviation Relationship Learning

<p align="center">
  <img src=./figs/model_workflow.jpg width="600"> 
</p>

This is a PyTorch implementation of the [TDD model](https://ieeexplore.ieee.org/abstract/document/10506667): 
```
@ARTICLE{10506667,
  author={Li, Jingtao and Wang, Xinyu and Wang, Shaoyu and Zhao, Hengwei and Zhong, Yanfei},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={One-Step Detection Paradigm for Hyperspectral Anomaly Detection via Spectral Deviation Relationship Learning}, 
  year={2024},
  volume={62},
  number={},
  pages={1-15},
  keywords={Task analysis;Hyperspectral imaging;Image reconstruction;Detectors;Training;Deep learning;Anomaly detection;Anomaly detection;deep learning;hyperspectral imagery (HSI);spectral deviation;unified model},
  doi={10.1109/TGRS.2024.3392189}}

@article{li2023one,
  title={One-Step Detection Paradigm for Hyperspectral Anomaly Detection via Spectral Deviation Relationship Learning},
  author={Li, Jingtao and Wang, Xinyu and Wang, Shaoyu and Zhao, Hengwei and Zhang, Liangpei and Zhong, Yanfei},
  journal={arXiv preprint arXiv:2303.12342},
  year={2023}
}
```

### Outline
1. In this paper, an unsupervised transferred direct detection (TDD) model is proposed, which is optimized directly for the anomaly detection task (**one-step paradigm**) and has **transferability**.
2. To train the TDD model in an unsupervised manner, **an anomaly sample simulation strategy** is proposed to generate numerous pairs of anomaly samples.
3. A **global self-attention module** and a **local self-attention module** are designed to help the model focus on the “spectrally deviating relationship.


### Introduction

Traditional deep detection models are optimized to complete a proxy task (two-step paradigm), such as background reconstruction or generation, rather than achieving anomaly detection directly. This leads to suboptimal results and poor transferability, which means that the deep model is trained and tested on the same image. In our work, an unsupervised transferred direct detection (TDD) model is proposed, which is optimized directly for the anomaly detection task (one-step paradigm) and has transferability. Specially, the TDD model is 
optimized to identify the spectral deviation relationship according to the anomaly definition. Compared to learning the specific 
background distribution as most models do, the spectral deviation relationship is universal for different images and guarantees the model transferability.

<p align="center">
  <img src=./figs/introduction_example.jpg width="600"> 
</p>

### Preparation

1. Install required packages according to the requirements.txt.
2. Download the four datasets and cropped patches from the following link and replace the `data` folder.
    (https://pan.baidu.com/s/12gDEvMjxxWIQ8IyQZPnZCQ?pwd=3z2v   password:3z2v)

### Model Training and Testing

1. TDD can be trained on any hyperspectral image and inferred directly on other images.
2. The training image needs to be cropped for patches with the script `utils/cropping.py`. For the HYDICE dataset in the paper, its patches have already be cropped in the given downloaded link.
3. Starting the training and testing process using the following command.
```
python run.py 'config_fie_path'
```
For example, to train the TDD model on the HYDICE dataset and infer on all four datasets.
```
python run.py ./configs/HYDICE/TDD_config.yaml
```

### Infer Only Without Labels

1. Modify the image paths and inferring settings (ckpt dir, infer patch size, etc.) in the Infer_config.yaml
```
python run.py './configs/Infer_config.yaml'
```



### Trained checkpoint

 &emsp;1. In practice, we found the model trained on the HYDICE dataset performed best. The trained checkpoint can be downloaded with the following link and put in the `ckpt` folder.
    (https://pan.baidu.com/s/12gDEvMjxxWIQ8IyQZPnZCQ?pwd=3z2v   password:3z2v)

 &emsp;2. Write the trained parameter path in the config file.
```
ckpt_dir: ./ckpts/
```

### Qualitative result  

 &emsp;The following are the qualitative results of TDD trained on the HYDICE dataset, with direct inference from the four datasets.

<p align="center">
  <img src=./figs/results_hydice.jpg width="600"> 
</p>

<p align="center">
  <img src=./figs/results_aviris.jpg width="600"> 
</p>

<p align="center">
  <img src=./figs/results_cri.jpg width="600"> 
</p>

<p align="center">
  <img src=./figs/results_river.jpg width="600"> 
</p>
