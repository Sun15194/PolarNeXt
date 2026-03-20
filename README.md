# **PolarNeXt: Rethink Instance Segmentation with Polar Representation**

The code for implementing the [**PolarNeXt**](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_PolarNeXt_Rethink_Instance_Segmentation_with_Polar_Representation_CVPR_2025_paper.html) (CVPR 2025). 


## News
- Our new polar-based model will be released soon: [Poly-DETR](https://github.com/Sun15194/Poly-DETR). Now, its corresponding paper "[Towards Instance Segmentation with Polygon Detection Transformers](https://arxiv.org/abs/2603.09245)" is available. (2026.03.20)
- Training code is uploaded. It supports training on dual NVIDIA RTX 4090D GPUs with about 16 hours. (2025.06.30)
- Validation code is updated. It supports inference at 49 FPS speed on a single NVIDIA RTX 4090D GPU. (2024.12.10)
- This work has been submitted to CVPR 2025. To ensure anonymity, all information regarding the authors and affiliations will remain undisclosed. (2024.11.15)


## Results
![Figure2](imgs/Figure2.jpg)

## Models

| Backbone | MS train | Lr schd | FPS  | AP<sup>val</sup> | AP<sup>test</sup> |        Weights         |
| :------: | :------: | :-----: | :--: | :--------------: | :---------------: | :--------------------: |
|   [R-50](projects/PolarNeXt/configs/polarnext_r50-torch_fpn_1x_coco.py)   |    N     |   1x    |  49  |       33.9       |         -         |           -            |
|   [R-50](projects/PolarNeXt/configs/polarnext_r50-torch_fpn_3x_ms_coco.py)   |    Y     |   3x    |  49  |       35.7       |       36.1        | [Baidu](https://pan.baidu.com/s/1LShE7EbeBsuK77I0hcSHbw?pwd=lyrn) & [Google](https://drive.google.com/file/d/10WT1sTK0nIQ_oWUCYFxxmplpw3RQiQ1J/view?usp=drive_link) |
|  [R-101](projects/PolarNeXt/configs/polarnext_r101-torch_fpn_3x_ms_coco.py)   |    Y     |   3x    |  38  |       37.1       |       37.4        | [Baidu](https://pan.baidu.com/s/1yXuZ01CQvnSeQNtzDOQYsA?pwd=lyrn) & [Google](https://drive.google.com/file/d/1WaB1vYC7YkUVZtQ23pc1_hJRU6qFx0Dq/view?usp=drive_link) |

- All models are trained on MS-COCO *train2017*.
- Data augmentation only contains random flip and scale jitter.
- For all 3x schedule settings, we default to using SGD to maintain fair comparison with other models. AdamW is recommended for 1x schedule, as using SGD with small batch size may lead to AP fluctuations of 0.2–0.4.

## Installation

Our PolarNeXt is fully based on [MMDetection](https://github.com/open-mmlab/mmdetection) **v3.3.0**. Please check [INSTALL.md](https://mmdetection.readthedocs.io/en/latest/get_started.html) for installation instructions.

#### Option 1: Install as a standalone project

You can install this project just like any standard MMDetection-based repository:

```bash
git clone https://github.com/Sun15194/PolarNeXt.git
cd PolarNeXt
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
pip install -v -e .
```

#### Option 2: Integrate into your own MMDetection codebase

You may also directly integrate our project into your existing MMDetection framework by following these steps:

1. Copy the entire `projects/` folder into your MMDetection root directory.
2. Overwrite the following files in your MMDetection repo with ours:
   - `mmdet/structures/mask/structures.py`
   - `setup.py` (from the root directory)
3. Run the following command in the root directory of your MMDetection repo to install the environment:

```bash
pip install -v -e .
```


To avoid potential errors, the following versions are recommanded:

```bash
Python == 3.10
PyTorch == 2.0.0
CUDA == 11.8
numpy == 1.24.3
matplotlib == 3.5.0
```
> ⚠️ **Note**: Make sure your MMDetection version is exactly **v3.3.0** to avoid compatibility issues.



## Run

(1) Training command:

- ```bash tools/dist_train.sh projects/PolarNeXt/configs/polarnext_r50-torch_fpn_3x_ms_coco.py --auto-scale-lr```

(2) Validation command:

- ```python tools/test.py projects/PolarNeXt/configs/polarnext_r50-torch_fpn_3x_ms_coco.py $YOUR_PTH_FILE ```

(3) Inference Speed command:

- ```python tools/analysis_tools/benchmark.py projects/PolarNeXt/configs/polarnext_r50-torch_fpn_3x_ms_coco.py --checkpoint $YOUR_PTH_FILE --task 'inference' ```

Notes: Considering that mask AP calculation requires converting polygons to pixel-level mask format, to ensure a fair comparison of inference speed, please comment out lines #600–603 in [`projects/PolarNeXt/model/head.py`](projects/PolarNeXt/model/head.py).



## Citation
If you use PolarNeXt in your research or wish to refer to the baseline results published here, please use the following BibTeX entry.

```
@InProceedings{Sun_2025_CVPR,
    author    = {Sun, Jiacheng and Zhou, Xinghong and Wu, Yiqiang and Zhu, Bin and Lu, Jiaxuan and Qin, Yu and Li, Xiaomao},
    title     = {PolarNeXt: Rethink Instance Segmentation with Polar Representation},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {19315-19324}
}
```

## Oi！
If you find this project helpful, please consider giving it a ⭐️ or a 🍴 on GitHub. Your support motivates us to keep improving!

