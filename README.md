# **PolarNeXt: Rethink Instance Segmentation with Polar Representation**

The code for implementing the **PolarNeXt**. 


## News
- Training code will also be uploaded soon.
- Test code is updated. It supports inference at 49 FPS speed on a single NVIDIA RTX 4090D GPU. (2024.12.10)
- This work has been submitted to CVPR 2025. To ensure anonymity, all information regarding the authors and affiliations will remain undisclosed. (2024.11.15)


## Results
![Figure2](imgs/Figure2.jpg)


## Models
| Backbone | MS train | Lr schd | FPS  | AP<sup>val</sup> | AP<sup>test</sup> | Weights |
| :------: | :------: | :-----: | :--: | :----: | :-----: | :-----: |
|   R-50   |    Y     |   3x    |  49  |  35.7  |  36.1   |  [model](www.baidu.com)  |
|  R-101   |    Y     |   3x    |  -   |  37.1  |  37.4   |  [model](www.baidu.com)  |

Notes:

- All models are trained on MS-COCO *train2017*.
- Data augmentation only contains random flip and scale jitter.


## Installation
Our PolarNeXt is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please check [INSTALL.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions.

## Testing
Inference testing command:

- ```python tools\test.py projects/PolarNeXt/configs/polarnext_r50_fpn_3x_coco.py checkpoints/polarnext_3x.pth --work-dir logs/polarnext-test```
