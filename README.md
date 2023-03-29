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

### SemanticKITTI test without TTA

| $\mathrm{PQ}$ | $\mathrm{PQ^{\dagger}}$ | $\mathrm{RQ}$ | $\mathrm{SQ}$ | $\mathrm{PQ}^{\mathrm{Th}}$ | $\mathrm{PQ}^{\mathrm{St}}$ | $\mathrm{mIoU}$ |
| :-------: | :-------: | :-------: | :-------: | :-------: | :-------: | :-------: |
| 64.9 | 70.0 | 75.9 | 84.9 | 67.1 | 63.3 | 68.3 |


## Code Release

The code is still going through large refactoring based on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d). Please stay tuned for the clean release.

## Citation

```bibtex
@article{xiao2023p3former,
    title={Position-Guided Point Cloud Panoptic Segmentation Transformer},
    author={Xiao, Zeqi and Zhang, Wenwei and Wang, Tai and Chen Change Loy and Lin, Dahua and Pang, Jiangmiao},
    journal={arXiv preprint},
    year={2023}
}
```

### Acknowledgements
We thank the contributors of [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) and the authors of [Cylinder3D](https://github.com/xinge008/Cylinder3D) for their great work, [K-Net](https://github.com/ZwwWayne/K-Net).
