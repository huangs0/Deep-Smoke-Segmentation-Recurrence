import torch.utils.data.dataloader
import DataSetLoader
import numpy as np

'''
PREDICT
PLEASE VERIFY THE DATASET, model.py and DataLoader.py is available before start training
Main Function:
    Set Training Parameters
    Load Model
    Load Data
    RUN THE NETWORK AND GET PREDICT DATASET
    Save PREDICT DATASET
'''

# Please Set the Predict Parameters here
batch_size = 5
# The average memory usage for 1 batch_size is 2GB, please refer to your gpu
use_cuda = torch.cuda.is_available()
# Recommend Using CUDA
print("Is CUDA available?", use_cuda)

print("Loading Model...")
Net = torch.load("BN_Avg_Res_Adammodel.pth")
'''
Available Net:
    Net = torch.load("BNmodel.pth")
    Net = torch.load("BN_Avg_Adammodel.pth")
    Net = torch.load("BN_Avg_Dense_Adammodel.pth")
    Net = torch.load("BN_Avg_Res_Adammodel.pth")


Copy and Paste  ：)
'''
print(Net)
if use_cuda:
    Net.cuda()
    # Convert model to cuda, type=FloatCudaTensor
print("Finishing Loading Model")

print("loading data....")
predict_data = DataSetLoader.PredictDataSet(inputpath="eval_input.npy")
#label data will not be used here
predict_loader = torch.utils.data.DataLoader(predict_data, batch_size=batch_size, shuffle=True, num_workers=5)
print("Finishing Loading Data, Starting Predicting")

predict = np.zeros([3000, 256, 256], dtype=np.uint8)

Net.eval()
batch_num = 0
for batch_idx, imgs in enumerate(predict_loader):
    print("Predicting batch_num:", batch_num)
    if use_cuda:
        imgs = imgs.cuda()
    out = Net(imgs)
    out = out.cpu().detach().numpy()
    # .cpu() transfer tensor from GPU to CPU, detach() give tensor out of grad
    for j in range(256):
        for k in range(256):
            # Compare two channel proabilities, remain the bigger one as final result
            # 1 represent smoke while 0 represent nonsmoke
            if out[0][0][j][k] <= out[0][1][j][k]:
                predict[batch_num * 5][j][k] = 1
            if out[1][0][j][k] <= out[1][1][j][k]:
                predict[batch_num * 5 + 1][j][k] = 1
            if out[2][0][j][k] <= out[2][1][j][k]:
                predict[batch_num * 5 + 2][j][k] = 1
            if out[3][0][j][k] <= out[3][1][j][k]:
                predict[batch_num * 5 + 3][j][k] = 1
            if out[4][0][j][k] <= out[4][1][j][k]:
                predict[batch_num * 5 + 4][j][k] = 1
    batch_num += 1
print("finishing predicting, start saving")
np.save("xxxxx", predict)
#Please Set your save file name here, .npy will be automatically added
print("Successfully saved")
