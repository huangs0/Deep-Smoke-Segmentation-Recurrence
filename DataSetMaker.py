import numpy
import os
import cv2

'''
THIS IS THE DATASET MAKER, NOT DATASET LOADER
DATA TYPE WILL BE numpy.uint8
MAIN FUNCTION:
    CONVERT THE .png IMAGE FILE INTO NUMPY ARRAY
    CONVERT THE [H,W,C] INTO [C,H,W]
    SAVE THE ARRAY INTO .npy NUMPY FILE
BECAUSE THE IMG FILE HAS BEEN RANDOMIZED, SO SIMPLY PICK 1-2800 as TRAINSET, 1-3000 as EVALUATESET    
'''

train_label = numpy.zeros((2800,256,256), dtype=numpy.uint8)
train_input = numpy.zeros((2800,3,256,256), dtype=numpy.uint8)
# Initialize the numpy array with the correct datatype

for i in range(1, 2801):
    # Indexstr is to find the corresponding .png file
    if i < 10:
        indexstr = "000" + str(i) + ".png"
    elif i < 100:
        indexstr = "00" + str(i) + ".png"
    elif i < 1000:
        indexstr = "0" + str(i) + ".png"
    else:
        indexstr = str(i) + ".png"
    lab = cv2.imread(os.path.join("target/", indexstr), cv2.IMREAD_GRAYSCALE)
    # target img has only GRAYSCALE channel
    inp = cv2.imread(os.path.join("input/", indexstr), cv2.IMREAD_UNCHANGED)
    for j in range(256):
        for k in range(256):
            train_input[i-1][0][j][k] = inp[j][k][0]
            train_input[i-1][1][j][k] = inp[j][k][1]
            train_input[i-1][2][j][k] = inp[j][k][2]
            # Here the input haven't been Normalized
            if lab[j][k] > 100:
                train_label[i-1][j][k] = 1
            else: train_label[i-1][j][k] = 0
            # If GRAYSCALE > 100, Recognize it as smoke, else no smoke

    print(i)
numpy.save("train_label", train_label)
numpy.save("train_input", train_input)

eval_label = numpy.zeros((3000,256,256), dtype=numpy.uint8)
eval_input = numpy.zeros((3000,3,256,256), dtype=numpy.uint8)

for i in range(1, 3001):
    # Indexstr is to find the corresponding .png file
    if i < 10:
        indexstr = "000" + str(i) + ".png"
    elif i < 100:
        indexstr = "00" + str(i) + ".png"
    elif i < 1000:
        indexstr = "0" + str(i) + ".png"
    else:
        indexstr = str(i) + ".png"
    lab = cv2.imread(os.path.join("target/", indexstr), cv2.IMREAD_GRAYSCALE)
    inp = cv2.imread(os.path.join("input/", indexstr), cv2.IMREAD_UNCHANGED)
    for j in range(256):
        for k in range(256):
            eval_input[i-1][0][j][k] = inp[j][k][0]
            eval_input[i-1][1][j][k] = inp[j][k][1]
            eval_input[i-1][2][j][k] = inp[j][k][2]
            if lab[j][k] > 100:
                eval_label[i-1][j][k] = 1
            else: eval_label[i-1][j][k] = 0
    print(i)
numpy.save("eval_label", eval_label)
numpy.save("eval_input", eval_input)
