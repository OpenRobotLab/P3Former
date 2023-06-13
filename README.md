# P3Former: Position-Guided Point Cloud Panoptic Segmentation Transformer

![main figure](https://user-images.githubusercontent.com/45515569/227226959-35f887e0-453b-4ac8-81c0-cb4b2f79333c.png)

## Introduction

This is an official release of [Position-Guided Point Cloud Panoptic Segmentation Transformer](https://arxiv.org/abs/2303.13509).


## Abstract

DEtection TRansformer (DETR) started a trend that uses a group of learnable queries for unified visual perception.
This work begins by applying this appealing paradigm to LiDAR-based point cloud segmentation and obtains a simple yet effective baseline.
Although the naive adaptation obtains fair results, the instance segmentation performance is noticeably inferior to previous works. 
By diving into the details, we observe that instances in the sparse point clouds are relatively small to the whole scene and often have similar geometry but lack distinctive appearance for segmentation, which are rare in the image domain. 
Considering instances in 3D are more featured by their positional information, we emphasize their roles during the modeling and design a robust Mixed-parameterized Positional Embedding (MPE) to guide the segmentation process. 
It is embedded into backbone features and later guides the mask prediction and query update processes iteratively, leading to Position-Aware Segmentation (PA-Seg) and Masked Focal Attention (MFA).
All these designs impel the queries to attend to specific regions and identify various instances. 
The method, named Position-guided Point cloud Panoptic segmentation transFormer (P3Former), outperforms previous state-of-the-art methods by 3.4% and 1.2% PQ on SemanticKITTI and nuScenes benchmark, respectively. 
The source code and models are available at https://github.com/SmartBot-PJLab/P3Former.



## Results

### SemanticKITTI test

| $\mathrm{PQ}$ | $\mathrm{PQ^{\dagger}}$ | $\mathrm{RQ}$ | $\mathrm{SQ}$ | $\mathrm{PQ}^{\mathrm{Th}}$ | $\mathrm{PQ}^{\mathrm{St}}$ | Download | Config |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| 65.3 | 67.8 | 74.9 | 86.6 | 67.4 | 63.7 | [model](https://drive.google.com/drive/folders/1RBDWV-oWOQsDAhNE8Z7SMLWsriLrpWCx?usp=sharing) | [config](https://github.com/SmartBot-PJLab/P3Former/blob/semantickitti/configs/p3former/p3former_8xb2_3x_semantickitti_trainval.py) |

### SemanticKITTI validation

| $\mathrm{PQ}$ | $\mathrm{PQ^{\dagger}}$ | $\mathrm{RQ}$ | $\mathrm{SQ}$ | $\mathrm{PQ}^{\mathrm{Th}}$ | $\mathrm{PQ}^{\mathrm{St}}$ | Download | Config |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| 62.6 | 66.2 | 72.4 | 76.2 | 69.4 | 57.7 | [model](https://drive.google.com/drive/folders/1RBDWV-oWOQsDAhNE8Z7SMLWsriLrpWCx?usp=sharing) | [config](https://github.com/SmartBot-PJLab/P3Former/blob/semantickitti/configs/p3former/p3former_8xb2_3x_semantickitti.py) |

* Pretraining a backbone is helpful to stablize the the training process and get slightly better results. You can pretrain a model with [config](https://github.com/SmartBot-PJLab/P3Former/blob/semantickitti/configs/cylinder3d/cylinder3d_8xb2_3x_semantickitti.py).

## Installation

Please follow [install.sh](https://github.com/SmartBot-PJLab/P3Former/blob/main/install.sh).

## Usage

### Data preparation

#### Semantickitti

```text
data/
├── semantickitti
│   ├── sequences
│   │   ├── 00
│   │   |   ├── labels
│   │   |   ├── velodyne
│   │   ├── 01
│   │   ├── ...
│   ├── semantickitti_infos_train.pkl
│   ├── semantickitti_infos_val.pkl
│   ├── semantickitti_infos_test.pkl

```

You can generate *.pkl by excuting

```
python tools/create_data.py semantickitti --root-path data/semantickitti --out-dir data/semantickitti --extra-tag semantickitti
```

## Training and testing

```bash
# train
sh dist_train.sh $CONFIG $GPUS

# val
sh dist_test.sh $CONFIG $CHECKPOINT $GPUS

# test
sh dist_test.sh $CONFIG $CHECKPOINT $GPUS

```

## Citation

```bibtex
@article{xiao2023p3former,
    title={Position-Guided Point Cloud Panoptic Segmentation Transformer},
    author={Xiao, Zeqi and Zhang, Wenwei and Wang, Tai and Chen Change Loy and Lin, Dahua and Pang, Jiangmiao},
    journal={arXiv preprint},
    year={2023}
}
```


## Acknowledgements

We thank the contributors of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and the authors of [Cylinder3D](https://github.com/xinge008/Cylinder3D) and [K-Net](https://github.com/ZwwWayne/K-Net) for their great work.
