import numpy
import torch.utils.data.dataset

'''
DATALOADER
FUNCTION:
    READ THE NUMPY ARRAY FILE AT INPUTPATH AND LABELPATH (MADE BY DataSetMaker.py)
    FIND THE ENTRY AND CORRESPONDING INDEX
    CONVERT THE ENTRY FROM numpy.uint8 INTO THE CORRESPONDING DATATYPE:
        input: FLoatTensor
        label: longTensor
    RETURN TO THE DATALOADER API OF torch.utils.data.dataloader    
'''

class SmokeDataSet(torch.utils.data.Dataset):
    # Overwrite the torch.utils.data.Dataset
    def __init__(self, inputpath, labelpath):
        self.input = numpy.load(inputpath)
        self.label = numpy.load(labelpath)

    def __len__(self):
        # overwrite the __len__
        return numpy.size(self.input, 0)

    def __getitem__(self, index):
        # overwrite the __getitem__
        inp = numpy.array(self.input[index], dtype=numpy.uint8)
        # find the corresponding entry and form new array with corresponding datatype
        lbl = numpy.array(self.label[index], dtype=numpy.uint8)
        return self.transform(inp, lbl)

    def transform(self, inp, lbl):
        inp = inp.astype(numpy.float64)
        # convert datatype
        inp = torch.from_numpy(inp).float()
        # create torch tensor
        lbl = torch.from_numpy(lbl).long()
        # label tensor must be datatype long
        return inp, lbl
