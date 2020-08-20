from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data.dataloader
import torchvision.models as models
from tensorboardX import SummaryWriter
import model
import DataSetLoader
import time

'''
TRAIN 
PLEASE VERIFY THE DATASET, model.py and DataLoader.py is available before start training
Main Function:
    Set Training Parameters
    Load Model
    Load Data
    Train the Network
    Use tensorboardX to visualize and track the training process
    Save Model
'''


# Please Set the Training Epoch here
batch_size = 7
# The average memory use is 1GB-1.5GB per batch, Please refer to your GPU set the batch_size, Remember left some memory for abundance
epoch_num = 51
# The average epoch_num to achieve nearly best performance is around 100
use_cuda = torch.cuda.is_available()
# Recommend Using Cuda
print("Is CUDA available?", use_cuda)

print("Loading Model...")
VGG = models.vgg16(pretrained=True)
# url = https://download.pytorch.org/models/vgg16-397923af.pth
Net = model.BNNet(VGG)
'''
Available Net:
    Net = model.OriginalNet(VGG)
    Net = model.BNNet(VGG)
    Net = model.BN_AvgNet(VGG)
    Net = model.BN_Avg_ResNet(VGG)
    Net = model.BN_Avg_DenseNet(VGG)
Copy and Paste XD
'''
print(Net)
writer = SummaryWriter(comment="xxxxxx")
# Initialize tensorboardX, please make sure commnet is correct before start training
if use_cuda:
    Net.cuda()
    # Convert model to cuda, type=FloatCudaTensor
print("Finishing Loading Model")

def criterion(output, label):
    loss = nn.CrossEntropyLoss()
    out = loss(output, label)
    return out

optimizer = optim.Adam(Net.parameters())
# optimizer = optim.SGD(Net.parameters(), lr = 0.001, momentum = 0.9, weight-decay = 1e-5)

print("loading data....")
train_data = DataSetLoader.SmokeDataSet(inputpath="train_input.npy", labelpath="train_label.npy")
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=5)
print("Finishing Loading Data, Starting Training")

def train(epoch, track):
    Net.train()
    # start the train model
    total_loss = 0
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        N = imgs.size(0)
        if use_cuda:
            imgs = imgs.cuda()
            labels = labels.cuda()
            # Convert it into GPU, type of imgs should be CUDAFloatTensor and label should be CUDALongTensor
        imgs_tensors = Variable(imgs)
        label_tensors = Variable(labels)
        out = Net(imgs_tensors)
        loss = criterion(out, label_tensors)
        loss /= N
        writer.add_scalar("loss", loss, global_step=1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (batch_idx) % 10 == 0:
            if track:
                # if track is true, track the filter feature map correctness
                for name, param in Net.named_parameters():
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), batch_idx)
            writer.add_scalar("avgloss", total_loss / (batch_idx + 1), global_step=1)
            print('train epoch [%d/%d], iter[%d/%d], aver_loss %.5f' % (epoch, epoch_num, batch_idx,
                                                                        len(train_loader),
                                                                        total_loss / (batch_idx + 1)))
    if (epoch) % 10 == 0:
        torch.save(Net, 'xxxxxxmodel%d.pth' % epoch)  # save for 5 epochs
        # Please make sure the file name is correct before training or previous file will be overwrited !!!
        total_loss /= len(train_loader)
        print('train epoch [%d/%d] average_loss %.5f' % (epoch, epoch_num, total_loss))


track = False
'''
This variable 'track' specifies whether tracking filter (weight and bias)
Please be careful to use this function because it will cost long time for I/O connection between CPU and GPU
And WILL SIGNIFICANTLY SLOW DOWN THE TRAINING SPEED BY 40% OR MORE
Only use this when you need to find errors in filter XD
'''
for epoch in range(epoch_num):
    start = time.time()
    train(epoch, track=track)
    end = time.time()
    writer.add_scalar("time", end - start, global_step=1)
writer.close()
