# **PolarNeXt: Rethink Instance Segmentation with Polar Representation**

The code for implementing the **PolarNeXt**. 


## News
- This work has been accepted by CVPR 2025. Training code will be uploaded after June 15th (CVPR conference).
- Test code is updated. It supports inference at 49 FPS speed on a single NVIDIA RTX 4090D GPU. (2024.12.10)
- This work has been submitted to CVPR 2025. To ensure anonymity, all information regarding the authors and affiliations will remain undisclosed. (2024.11.15)


## Results
![Figure2](imgs/Figure2.jpg)


## Models
| Backbone | MS train | Lr schd | FPS  | AP<sup>val</sup> | AP<sup>test</sup> | Weights |
| :------: | :------: | :-----: | :--: | :----: | :-----: | :-----: |
|   R-50   |    Y     |   3x    |  49  |  35.7  |  36.1   |  [model](https://pan.baidu.com/s/1LShE7EbeBsuK77I0hcSHbw?pwd=lyrn)  |
|  R-101   |    Y     |   3x    |  38  |  37.1  |  37.4   |  [model](https://pan.baidu.com/s/1yXuZ01CQvnSeQNtzDOQYsA?pwd=lyrn)  |

Notes:

- All models are trained on MS-COCO *train2017*.
- Data augmentation only contains random flip and scale jitter.


## Installation
Our PolarNeXt is based on [mmdetection](https://github.com/open-mmlab/mmdetection). Please check [INSTALL.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions.

## Testing
Test commands:
- ```python tools\test.py projects/PolarNeXt/configs/polarnext_r50_fpn_3x_coco.py checkpoints/polarnext_r50_epoch_36.pth --work-dir logs/polarnext-r50-test```
- ```python tools\test.py projects/PolarNeXt/configs/polarnext_r50_fpn_3x_coco.py checkpoints/polarnext_r101_epoch_36.pth --work-dir logs/polarnext-r101-test```
  
Inference Speed command:
- ```python tools\analysis_tools\benchmark.py projects/PolarNeXt/configs/polarnext_r50-torch_fpn_3x_ms_coco.py --checkpoint checkpoints/polarnext-r50-3x/epoch_36.pth --task inference --work-dir logs/polarnext-r50-benchmark```
- ```python tools\analysis_tools\benchmark.py projects/PolarNeXt/configs/polarnext_r101-torch_fpn_3x_ms_coco.py --checkpoint checkpoints/polarnext-r101-3x/epoch_36.pth --task inference --work-dir logs/polarnext-r101-benchmark```

**Note:** To compute the mask AP, the polygons detected by PolarNeXt will be converted into mask format. Therefore, for a fair comparison, please comment out lines 257–260 in [`projects/PolarNeXt/model/head.py`](projects/PolarNeXt/model/head.py) when running the inference speed command.
