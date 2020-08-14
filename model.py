import torch
import torch.nn as nn
import torch.nn.functional as F

'''
MODEL LOADER
MAIN FUNCTION:
    RECORD THE MODELS
    INITIALIZE AND RETURN MODELS
!!! IMPORTANT !!!
ALL THE MODELS MUST BE INITIALIZED WITH VGG16 MODEL, PLEASE USE VGG AS PARAMETER "model" TO THE INITIALIZER
!!! IMPORTANT !!!
'''

class OriginalNet(nn.Module):
    # All Model Class overwrite the nn.Module class
    def __init__(self, model):
        super(OriginalNet, self).__init__()
        self.conv1 = nn.Sequential(*list(model.children())[0][0 : 3])
        #First skip, connect with decodey2
        self.conv2 = nn.Sequential(*list(model.children())[0][4 : 8])
        #Second Skip, connect with decodey1
        self.conv3 = nn.Sequential(*list(model.children())[0][9 : 15])
        # Third Skip, connect with decodex2
        self.conv4 = nn.Sequential(*list(model.children())[0][16 : 22])
        #Forth Skip, connect with decodex1
        self.conv5 = nn.Sequential(*list(model.children())[0][23 : 28])
        # The twopath structure are named with x and y, where x refer to first path and y refer to second path
        self.decodey1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        self.decodey2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.decodey3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decodex1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex3 = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        skip1 = self.conv1(x)
        skip2 = self.conv2(skip1)
        skip3 = self.conv3(skip2)
        skip4 = self.conv4(skip3)
        out = self.conv5(skip4)
        x = self.decodex1(out)
        x = torch.cat((x, skip4), dim=1)
        x = self.decodex2(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.decodex3(x)
        y = self.decodey1(skip3)
        y = torch.cat((y, skip2), dim=1)
        y = self.decodey2(y)
        y = torch.cat((y, skip1), dim=1)
        y = self.decodey3(y)
        x = torch.add(x,y)
        x = self.decode(x)
        return x

class BNNet(nn.Module):
    # All Model Class overwrite the nn.Module class
    def __init__(self, model):
        super(BNNet, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Sequential(*list(model.children())[0][0 : 3])
        self.bn2 = nn.BatchNorm2d(64)
        #First skip, connect with decodey2
        self.conv2 = nn.Sequential(*list(model.children())[0][4 : 8])
        self.bn3 = nn.BatchNorm2d(128)
        #Second Skip, connect with decodey1
        self.conv3 = nn.Sequential(*list(model.children())[0][9 : 15])
        self.bn4 = nn.BatchNorm2d(256)
        # Third Skip, connect with decodex2
        self.conv4 = nn.Sequential(*list(model.children())[0][16 : 22])
        self.bn5 = nn.BatchNorm2d(512)
        #Forth Skip, connect with decodex1
        self.conv5 = nn.Sequential(*list(model.children())[0][23 : 28])
        self.bn6 = nn.BatchNorm2d(512)
        # The twopath structure are named with x and y, where x refer to first path and y refer to second path
        self.decodey1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        self.decodey2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.decodey3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decodex1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex3 = nn.Sequential(
            nn.BatchNorm2d(768),
            nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        skip1 = self.bn2(out)
        out = self.conv2(skip1)
        skip2 = self.bn3(out)
        out = self.conv3(skip2)
        skip3 = self.bn4(out)
        out = self.conv4(skip3)
        skip4 = self.bn5(out)
        out = self.conv5(skip4)
        out = self.bn6(out)
        x = self.decodex1(out)
        x = torch.cat((x, skip4), dim=1)
        x = self.decodex2(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.decodex3(x)
        y = self.decodey1(skip3)
        y = torch.cat((y, skip2), dim=1)
        y = self.decodey2(y)
        y = torch.cat((y, skip1), dim=1)
        y = self.decodey3(y)
        x = torch.add(x, y)
        x = self.decode(x)
        return x

class BN_AvgNet(nn.Module):
    # All Model Class overwrite the nn.Module class
    def __init__(self, model):
        super(BN_AvgNet, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Sequential(*list(model.children())[0][0 : 3])
        self.bn2 = nn.BatchNorm2d(64)
        #First skip, connect with decodey2
        self.conv2 = nn.Sequential(*list(model.children())[0][5 : 8])
        self.bn3 = nn.BatchNorm2d(128)
        #Second Skip, connect with decodey1
        self.conv3 = nn.Sequential(*list(model.children())[0][10 : 15])
        self.bn4 = nn.BatchNorm2d(256)
        # Third Skip, connect with decodex2
        self.conv4 = nn.Sequential(*list(model.children())[0][17 : 22])
        self.bn5 = nn.BatchNorm2d(512)
        #Forth Skip, connect with decodex1
        self.conv5 = nn.Sequential(*list(model.children())[0][24 : 28])
        self.bn6 = nn.BatchNorm2d(512)
        # The twopath structure are named with x and y, where x refer to first path and y refer to second path
        self.decodey1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        self.decodey2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.decodey3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decodex1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex3 = nn.Sequential(
            nn.BatchNorm2d(768),
            nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        skip1 = self.bn2(out)
        out = self.conv2(F.normalize(self.downsample(skip1)))
        skip2 = self.bn3(out)
        out = self.conv3(F.normalize(self.downsample(skip2)))
        skip3 = self.bn4(out)
        out = self.conv4(F.normalize(self.downsample(skip3)))
        skip4 = self.bn5(out)
        out = self.conv5(F.normalize(self.downsample(skip4)))
        out = self.bn6(out)
        x = self.decodex1(out)
        x = torch.cat((x, skip4), dim=1)
        x = self.decodex2(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.decodex3(x)
        y = self.decodey1(skip3)
        y = torch.cat((y, skip2), dim=1)
        y = self.decodey2(y)
        y = torch.cat((y, skip1), dim=1)
        y = self.decodey3(y)
        x = torch.add(x, y)
        x = self.decode(x)
        return x

class BN_Avg_ResNet(nn.Module):
    # All Model Class overwrite the nn.Module class
    def __init__(self, model):
        super(BN_Avg_ResNet, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Sequential(*list(model.children())[0][0 : 3])
        self.bn2 = nn.BatchNorm2d(64)
        #First skip, connect with decodey2, AND FIRST RESIDUAL
        self.conv2 = nn.Sequential(*list(model.children())[0][5 : 8])
        self.bn3 = nn.BatchNorm2d(128)
        #Second Skip, connect with decodey1, first add
        self.selfconv1 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        #Second Residual
        self.conv3 = nn.Sequential(*list(model.children())[0][10 : 15])
        self.bn4 = nn.BatchNorm2d(256)
        # Third Skip, connect with decodex2, second add
        self.selfconv2 = nn.Conv2d(128, 256, kernel_size=(1, 1), stride=(1, 1))
        #Third Residual
        self.conv4 = nn.Sequential(*list(model.children())[0][17 : 22])
        self.bn5 = nn.BatchNorm2d(512)
        #Forth Skip, connect with decodex1, third add
        self.selfconv3 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1))
        #Fourth Residual
        self.conv5 = nn.Sequential(*list(model.children())[0][24 : 28])
        #Fourth add
        self.bn6 = nn.BatchNorm2d(512)
        # The twopath structure are named with x and y, where x refer to first path and y refer to second path
        self.decodey1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        self.decodey2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.decodey3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decodex1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex3 = nn.Sequential(
            nn.BatchNorm2d(768),
            nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        skip1 = self.bn2(out)
        out = self.conv2(F.normalize(self.downsample(skip1)))
        skip2 = self.bn3(out)
        res2 = torch.add(self.selfconv1(self.downsample(skip1)), skip2)
        out = self.conv3(F.normalize(self.downsample(res2)))
        skip3 = self.bn4(out)
        res3 = torch.add(self.selfconv2(self.downsample(res2)), skip3)
        out = self.conv4(F.normalize(self.downsample(res3)))
        skip4 = self.bn5(out)
        res4 = torch.add(self.selfconv3(self.downsample(res3)), skip4)
        out = self.conv5(F.normalize(self.downsample(res4)))
        out = self.bn6(out)
        out = torch.add(self.downsample(res4), out)
        x = self.decodex1(out)
        x = torch.cat((x, skip4), dim=1)
        x = self.decodex2(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.decodex3(x)
        y = self.decodey1(skip3)
        y = torch.cat((y, skip2), dim=1)
        y = self.decodey2(y)
        y = torch.cat((y, skip1), dim=1)
        y = self.decodey3(y)
        x = torch.add(x, y)
        x = self.decode(x)
        return x

class BN_Avg_DenseNet(nn.Module):
    # All Model Class overwrite the nn.Module class
    def __init__(self, model):
        super(BN_Avg_DenseNet, self).__init__()
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.conv1 = nn.Sequential(*list(model.children())[0][0 : 3])
        self.bn2 = nn.BatchNorm2d(64)
        #First skip, connect with decodey2, AND FIRST RESIDUAL
        self.conv2 = nn.Sequential(*list(model.children())[0][5 : 8])
        self.bn3 = nn.BatchNorm2d(128)
        #Second Skip, connect with decodey1, first add
        self.selfconv1 = nn.Conv2d(192, 128, kernel_size=(1, 1), stride=(1, 1))
        #Second Residual
        self.conv3 = nn.Sequential(*list(model.children())[0][10 : 15])
        self.bn4 = nn.BatchNorm2d(256)
        # Third Skip, connect with deodex2, second add
        self.selfconv2 = nn.Conv2d(384, 256, kernel_size=(1, 1), stride=(1, 1))
        #Third Residual
        self.conv4 = nn.Sequential(*list(model.children())[0][17 : 22])
        self.bn5 = nn.BatchNorm2d(512)
        #Forth Skip, connect with decodex1, third add
        self.selfconv3 =nn.Conv2d(768, 512, kernel_size=(1, 1), stride=(1, 1))
        #Fourth Residual
        self.conv5 = nn.Sequential(*list(model.children())[0][24 : 28])
        #Fourth add
        self.bn6 = nn.BatchNorm2d(512)
        self.selfconv4 = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1))
        # The twopath structure are named with x and y, where x refer to first path and y refer to second path
        self.decodey1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )

        self.decodey2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.decodey3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decodex1 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(512),

            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
        )
        self.decodex3 = nn.Sequential(
            nn.BatchNorm2d(768),
            nn.Conv2d(768, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 2, kernel_size=(1, 1), stride=(1, 1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.bn1(x)
        out = self.conv1(out)
        skip1 = self.bn2(out)
        out = self.conv2(F.normalize(self.downsample(skip1)))
        skip2 = self.bn3(out)
        out = torch.cat((self.downsample(skip1), skip2), dim=1)
        res2 = self.selfconv1(out)
        out = self.conv3(F.normalize(self.downsample(res2)))
        skip3 = self.bn4(out)
        out = torch.cat((self.downsample(res2), skip3), dim=1)
        res3 = self.selfconv2(out)
        out = self.conv4(F.normalize(self.downsample(res3)))
        skip4 = self.bn5(out)
        out = torch.cat((self.downsample(res3), skip4), dim=1)
        res4 = self.selfconv3(out)
        out = self.conv5(F.normalize(self.downsample(res4)))
        out = self.bn6(out)
        out = torch.cat((self.downsample(res4),out), dim=1)
        out = self.selfconv4(out)
        x = self.decodex1(out)
        x = torch.cat((x, skip4), dim=1)
        x = self.decodex2(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.decodex3(x)
        y = self.decodey1(skip3)
        y = torch.cat((y, skip2), dim=1)
        y = self.decodey2(y)
        y = torch.cat((y, skip1), dim=1)
        y = self.decodey3(y)
        x = torch.add(x, y)
        x = self.decode(x)
        return x
