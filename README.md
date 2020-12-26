# Deep-Smoke-Segmentation-Recurrence
"Deep Smoke Segmentation" Paper Recurrence and Optimization

Collaborators:
Huang Songlin, Student, Department of Computer Science, Faculty of Engineering, HKU
Zheng Guang, Vice Professor, Faculty of EEE, HQU

Please note that this project has been finished and no longer maintained, but if you encouter any problem, please feel free to email me at huangs0@hku.hk


## Papers being referenced in this project
Original Paper "Deep Smoke Segmentation": https://arxiv.org/pdf/1809.00774

ResNet: https://arxiv.org/abs/1512.03385

Batch Normalize:  https://arxiv.org/pdf/1502.03167

Adam: https://arxiv.org/pdf/1412.6980

FCN: https://arxiv.org/abs/1411.4038

DenseNet: https://arxiv.org/pdf/1608.06993

SSD Optimization: https://arxiv.org/pdf/1611.10012

SSD: https://arxiv.org/pdf/1512.02325

## Environment:
#### RTX2080Ti 11GB GPU
#### i7-4790 4-core CPU
#### Ubuntu 18.04.3 LTS
#### Driver Version 440.100
#### CUDA 10.2
#### cuDNN 8.02
#### Pytorch 1.3.0
#### Numpy 1.17.3
#### opencv-python 4.1.1.26
#### tensorboardX 2.1 with tensorboard 2.3.0
## Additional Packages
#### tensorboardX is used for tracking parameters in Net, see https://github.com/lanpa/tensorboardX
#### thop is used to calculating flops and parameter amount for comparative study, 'pip install thop'

## Repository Structure: For Details Please search python files
DataSetMaker is used to convert image files (Format need to be supported by opencv-python) into the numpy array file for DataSetLoader

DataSetLoader overwrite torch.utils.data.dataloader, turn the numpy array from DataSetMaker into array for train.py or predict.py

model overwrite torch.nn.module, record model details and left API for client in train to initialize model

train: main function, load the DataSet, initialize model, set trainning parameter, train, record data use tensorboard and save model

predict: load DataSet, set predict parameter, predict the model, save predict result into numpy array

eval: load numpy array recording predict details, evaluate model performance using mIoU and mMse

For dataset, they can be download from author's official website of original paper 《Deep Smoke Segmentation》
Please see http://staff.ustc.edu.cn/~yfn/dss.html for more details

## Introduction

Model from Deep Smoke Segmentation is the optimal from FCN-8s, following the traditional Encoder-Decoder Structure and skip structure can also be found in it

But unlike original FCN-8s, this model create a two path structure for decoder, first path, named x, start from the end of encoder, i.e. the end of fifth convolution and downsample layers while the second path, named y, start from the middleware of encoder, more exactly, the end of third convolutional layer. At the end of decoding, two path will be combine together by using add operation

Further more, rather than using the deconvolution in FCN, this paper replace it to the UpSamplingBilinear2d with another convolution layer without change pixel size, it argues this can improve the accuracy in classification

Details provided by paper are as follows:

### Loss Function: Cross Entropy Loss with L2 Norm
### Optimizer: SGD with lr = 0.0001, momentum = 0.9, weight_decay = 1e-5
### Measurement: mIOU and mMse

## Other 

## Optimization

#### Introducing the BatchNormal Layer
