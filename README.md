# Deep-Smoke-Segmentation-Recurrence
"Deep Smoke Segmentation" Paper Recurrence and Optimization

If you like this repository, or it helps you a lot, that will be my honor. Please give me a star at this repository, thank you!

## For Recurrence or Optimization Details, please see 
https://mp.weixin.qq.com/s__biz=MzkyNDA0OTIzNQ==&mid=2247483692&idx=1&sn=c195da2f27f228797dc47230416acf3a&chksm=c1da8b10f6ad02063bde9f1456bacd205d43bf3864ff5fe0a89435893bd59971ddecfd2c1fa1&token=116246471&lang=zh_CN#rd
## Want the papers mentioned in above article, please see:
SSD Optimization: https://arxiv.org/pdf/1611.10012

SSD: https://arxiv.org/pdf/1512.02325

Original Paper "Deep Smoke Segmentation": https://arxiv.org/pdf/1809.00774

ResNet: https://arxiv.org/abs/1512.03385

Batch Normalize:  https://arxiv.org/pdf/1502.03167

Adam: https://arxiv.org/pdf/1412.6980

FCN: https://arxiv.org/abs/1411.4038

DenseNet: https://arxiv.org/pdf/1608.06993

## Environment:
#### RTX2080Ti 11GB
#### i7-4790 4-core
#### Ubuntu 18.04.3 LTS
#### Driver Version 440.100
#### CUDA 10.2
#### cuDNN 8.02
#### Pytorch 1.3.0
#### Numpy 1.17.3
#### opencv-python 4.1.1.26
#### tensorboardX 2.1 with tensorboard 2.3.0

## Repository Structure: For Details Please search python files
DataSetMaker is used to convert image files (Format need to be supported by opencv-python) into the numpy array file for DataSetLoader

DataSetLoader overwrite torch.utils.data.dataloader, turn the numpy array from DataSetMaker into array for train.py or predict.py

model overwrite torch.nn.module, record model details and left API for client in train to initialize model

train: main function, load the DataSet, initialize model, set trainning parameter, train, record data use tensorboard and save model

predict: load DataSet, set predict parameter, predict the model, save predict result into numpy array

eval: load numpy array recording predict details, evaluate model performance using mIoU and mMse

## Some Q&A:

Why use Numpy? For 3,000 Images, each runtime dataloader need to read by cv2, convert [H,W,C] into [C,H,W], this is too much time consuming so I only run it one time by DataSetMaker.py and save the result into .npy files
