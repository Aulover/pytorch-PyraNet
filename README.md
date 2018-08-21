# Pytorch-PyraNet

## Introduction
This is a pytorch version code of 'Learning Feature Pyramids for Human Pose Estimation', ICCV 2017. Link of paper: https://arxiv.org/pdf/1708.01101.pdf.

## The network
The network designed based on stacked hourglass network. 

Pyranet replace the Residual module with Pyramid Residual module. The network:
![avatar](https://github.com/IcewineChen/pytorch-PyraNet/blob/master/imgs/network.png)

The authors design a Pyramid Residual module to use features and information of multi-scale:
![avatar](https://github.com/IcewineChen/pytorch-PyraNet/blob/master/imgs/prm.png)

In this repo, PRM-A, PRM-B and PRM-C has been realized. Following the comment you can choose these three Pyramid Residual Module. 
See the definition of Pyramid Residual Module in models/prm.py and the network architecture in models/network.py. 

## Requirements
The code has been tested with Ubuntu 16.04 and CUDA 8.

- python 2.7
- pytorch == 0.4.1  (must be pytorch 0.4.1)
- opencv-python
- numpy
- progress

## Datasets
Mpii human pose dataset. Details of path setting: see ref.py. You can set the path in this file.

## Train
For example, if you train the PRM network with 300 epochs, batch 6 and 2 stacked hourglass module with one gpu:
```
CUDA_VISIBLE_DEVICES = 0 python main.py -nEpochs 300 -trainBatch 6 -nStack 2
```
If you have a multi-gpu server, you can uncomment line27 in train.py to get a parallel speed-up:
```
output = torch.nn.parallel.data_parallel(model,input_var,device_ids=[0,1,2,3,4,5])
``` 
and then set CUDA_VISIBLE_DEVICES=0,1,2,3,4,5.

Finally, you can see more usage of flags in opts.py, such as -expID for specifing the path to save models and predictions.

## Tools
network_visual.py : Make network architecture visualization

tools/eval_pckh.py : Get the result of pckh@0.x

## Evaluation

Result: using tools/eval_pckh.py for evaluation.You can get a result like that(after 160 epochs):
```
Model,  Head,   Shoulder, Elbow,  Wrist,   Hip,     Knee,    Ankle,  Mean
hg      96.69   95.06     88.38   83.30    86.31    82.81    78.86   87.43
```

## Acknowledgement
Thanks for the authors of 'Learning Feature Pyramids for Human Pose Estimation'.
