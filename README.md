# SPA
![](https://img.shields.io/github/license/huoxiangzuo/SPA)  
This repo. is the official implementation of ['**SPA: Self-Peripheral-Attention for Central-Peripheral Interactions in Endoscopic Image Classification and Segmentation**'](https://www.sciencedirect.com/science/article/pii/S0957417423035558).   
Authors: Xiangzuo Huo, Shengwei Tian, Yongxu Yang, Long Yu, Wendong Zhang, Aolun Li.  
Enjoy the code and find its convenience to produce more awesome works!

## Motivation
<img src="https://github.com/huoxiangzuo/SPA/assets/57312968/6ae2a1e0-bb96-4152-bf63-a44b8c1ed653" width="500">

## SPA-Net
![spanet](https://github.com/huoxiangzuo/SPA/assets/57312968/ecf8c9c3-5e47-434a-8a69-2a2339ced0b2)

## Self-Peripheral-Attention
![SPAnorm](https://github.com/huoxiangzuo/SPA/assets/57312968/49c5f7d7-2c25-4597-80dc-859b0aa5fb88)

## Visual Inspection of SPA
<img src="https://github.com/huoxiangzuo/SPA/assets/57312968/84bce289-e6dc-402e-98ab-f61c9bfe77ee" width="500">

## Segmentation Results
<img src="https://github.com/huoxiangzuo/SPA/assets/57312968/7707fe7b-2d2d-4cb3-949d-0ee5c548738e" width="800">

## Run
0. Requirements:
* python3
* pytorch 1.12.0
* torchvision 0.13.0
1. Train:
* Prepare the required images and store them in categories, set up training image folders and validation image folders respectively
* Run `python train.py`
2. Resume train:
* Modify `parser.add_argument('--RESUME', type=bool, default=True)` in `python train.py`
* Run `python train.py`
3. Test:
* Run `python test.py`
4. Predict:
* Run `python predict.py`

## TensorBoard
Run `tensorboard --logdir runs --port 6006` to view training progress

## Reference
Some of the codes in this repo are borrowed from:  
* [SwinTransformer](https://github.com/microsoft/Swin-Transformer)  
* [FocalNet](https://github.com/microsoft/FocalNet) 
* [WZMIAOMIAO](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing)

## Citation

If you find our paper/code is helpful, please consider citing:

```bibtex
@article{huo2023spa,
  title={SPA: Self-Peripheral-Attention for central-peripheral interactions in endoscopic image classification and segmentation},
  author={Huo, Xiangzuo and Tian, Shengwei and Yang, Yongxu and Yu, Long and Zhang, Wendong and Li, Aolun},
  journal={Expert Systems with Applications},
  pages={123053},
  year={2023},
  publisher={Elsevier}
}
```

