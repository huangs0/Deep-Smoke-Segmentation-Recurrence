import numpy as np

'''
EVAL 
PLEASE VERIFY THE EVAL DATASET, .pth model file and DataLoader.py is available before start evaluate
Main Function:
    Load Predict DataSet and Predict Label DataSet
    Calculate the Confuse Metrics
    Calculate mIOU and mMse
'''

class Eval(object):
    def __init__(self):
        self.confuse_metrics = np.zeros((2,2))

    def get_confuse_metrics(self, target, predict):
        mask = (target >= 0) & (target < 2)
        label = 2 * target[mask].astype('int') + predict[mask]
        count = np.bincount(label, minlength=2 ** 2)
        confusion_metrics = count.reshape(2, 2)
        return confusion_metrics

    def add(self, target, predict):
        self.confuse_metrics += self.get_confuse_metrics(target, predict)

    def mIoU(self):
        IoU = np.diag(self.confuse_metrics) / (
                np.sum(self.confuse_metrics, axis=1) + np.sum(self.confuse_metrics, axis=0) -
                np.diag(self.confuse_metrics))
        MIoU = np.nanmean(IoU)
        return MIoU

    def Mse(self, target, predict):
        Mse = 0
        for i in range(256):
            for j in range(256):
                Mse += np.square(int(predict[i][j]) - int(target[i][j]))
        return Mse / 65536

def eval(predict, label, batch_size):
    mIoU = np.zeros(batch_size, dtype=np.float64)
    mMse = np.zeros(batch_size, dtype=np.float64)
    obj = Eval()
    print("Start Evaluating")
    for batch_idx in range(batch_size):
        print("Evaluating Batch:", batch_idx)
        obj.add(target=label[batch_idx], predict=predict[batch_idx])
        mIoU[batch_idx] = obj.mIoU()
        mMse[batch_idx] = obj.Mse(target=label[batch_idx],predict=predict[batch_idx])
    print("Finishing Evaluating, Returning")
    print(mMse)
    print("mMse=", np.sum(mMse) / 3000)
    print(mIoU)
    print("mIoU=", np.sum(mIoU) / 3000)
    return mMse, mIoU



predict = np.load("xxxx.npy")
#Please set your predict.py here
label = np.load("eval_label.npy")
batch_size = 3000
#Please set your batch size 
eval(predict=predict, label=label, batch_size=batch_size)
