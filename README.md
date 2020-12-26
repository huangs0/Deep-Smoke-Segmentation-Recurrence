# Deep-Smoke-Segmentation-Recurrence
"Deep Smoke Segmentation" Paper Recurrence and Optimization

Huang Songlin, Student, Department of Computer Science, Faculty of Engineering, HKU

### Instructor:
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

Following Figures can be found at: 

#### Introducing the BatchNormal Layer
```Python
nn.BatchNorm2d(channel)
```
Original model lacks BatchNormal operation while it's a model with more than 60 layers and this will cause the trainning become nearly impossible, i.e. if you track the layer parameters distribution and use histogram of tensorboardX to print results, then results will show no distribution change and loss function will keep in high frequency oscillatory with no decrease. Google's 《Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift》 provide a simple but efficient solution -- adding Batch Normal Operation after ReLU. By adding BN layer, parameter distribution start moving while loss function starts decline to around 0.6

### Replace the MaxPool Layer with AvgPool
```Python
nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
# change to
nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
```

By comparing the filter before and after pool layer, we found that value of tensor change very much, due to the MaxPool Layer. While now the Network's Loss function are in very big oscillatory with slow decline. We guess whether or nnot the vibration is partially due to the Pool layer as the MaxPool only remain the value of biggest point among the square. The SSD3000 model remind us an alternative of MaxPool, the AvgPool Operation which take the average value among the square. After change the MaxPool to AvgPool, the voladity of loss function, has declined, while as the price, time cost for training per epoch has increased a little

### Replace Optimizer to Adam
```Python
optim.SGD(Net.parameters(), lr = 0.001, momentum = 0.9, weight-decay = 1e-5)
#to
optim.Adam(Net.parameters())
```
As there's many possibility from many papers to test, the trainning speed of SGD is lower than our expectation. It needs around 50 epoch to approach the best performance in theory, which will take near 1.5hr in practice. And we want to test the performance first and then train the final model using SGD to get the actually best performance So we try to replace it with Adam, which is proved can make Nerual Network approach the theorical best performance faster than SGD. It works and save around 50% epoch to approach the point. As the price, voladity has increased. 

Special Note: When using Adam, PLEASE DON'T ADD THE L2Norm. This will cause the dying ReLU problem, i.e. the distribution of parameter will flying into around 0. Then according to definition of ReLU, if parameter > 0, f(x)=x, there'll be no path for parameter to escape from 0. Even worse. if x < 0, then f(x) = 0, the parameter become totally dead. We do some research and find the major reason is because the L2Norm, L2Norm will decrease the possible value domain of parameter x and this side effect is very significant in Adam optimizer, inner mechanism is still unclear.

### Adding Shortcut Structure
Previously what we have done will not actually improve the best performance of model in practice, just improve the trainning speed, reduce voladity. Later, we focus on how we can improve the Network performance actually. ResNet introduce a simple but efficient way for Deep Nerual Network to improve the performance by adding a shortcut structure and DenseNet goes further and introduce the way by adding catch to make later layer get every information from the previous layer. Following their path we add the shortcut using add or catch operation. As a straight but not such accurate way, we compare the loss function before and after adding the layer and between adding add and catch layers. Result is when adding shortcut by add operation, loss function will go down to nearly 4.5 comparing to 5.5 before adding shortcut. In the meanwhile, adding shortcut by concat operation, loss function will go down to around 4.8 but cause the vibration increase and unsufficient change in parameter distribution.

## Final Result
Finally we choose the Adam, AvgPool, shortcut with add operation, BatchNormal Layer as final test operation.
As comparison, MIoU and mMse provided by paper is 0.7029, 0.2833 for original model
In final version, after trainning the first epoch, MIoU = 0.7523 while mMse = 0.1052
After 200 epoch, MIoU = 0.9854, mMse = 0.0075
There's a big improve in the evaluation results

# More Information
For more information, please refer to https://mp.weixin.qq.com/s__biz=MzkyNDA0OTIzNQ==&mid=2247483692&idx=1&sn=c195da2f27f228797dc47230416acf3a&chksm=c1da8b10f6ad02063bde9f1456bacd205d43bf3864ff5fe0a89435893bd59971ddecfd2c1fa1&token=116246471&lang=zh_CN#rd 
This is my personal blog
