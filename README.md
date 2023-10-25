# SPA
![](https://img.shields.io/github/license/huoxiangzuo/SPA)  
This repo. is the official implementation of '**SPA: Self-Peripheral-Attention for Central-Peripheral Interactions in Endoscopic Image Classification and Segmentation**'.   
Authors: Xiangzuo Huo, Shengwei Tian, Yongxu Yang, Long Yu, Wendong Zhang, Aolun Li.  
Enjoy the code and find its convenience to produce more awesome works!

## Overview
<!-- <img width="1395" alt="figure1" src="https://user-images.githubusercontent.com/57312968/191570017-34f30c13-9d8e-4776-a118-de968aebdb19.png" width="80%"> -->

## SPA-Net and Self-Peripheral-Attention
<!-- <img width="1424" alt="figure2s" src="https://user-images.githubusercontent.com/57312968/191570496-c62e04dc-8baf-4b01-a6ba-03c24c5a744d.png" width="70%"> -->

## Visual Inspection of SPA
<!-- <img src="https://user-images.githubusercontent.com/57312968/191570242-4425944d-4017-45c6-a3f7-f977376766a2.png" width="75%"> -->

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

```

