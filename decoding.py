# python 3.9.5
from math import ceil
import cv2


import numpy as np

from functions import *
from convert_audio_to_2d import *
windowSize=8
QTY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  #  quantization table
                [12, 12, 14, 19, 26, 48, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]])

QTC = np.array([  [1, 1, 1, 1, 2, 4, 1, 1],
                [1, 17, 17, 17, 2, 5, 17, 5],
                [17, 17, 17, 2, 4, 5, 9, 6],
                [1, 1, 2, 7, 5, 8, 1, 7],
                [1, 2, 3, 5, 6, 19, 1, 7],
                [7, 3, 5, 6, 8, 14, 1, 2],
                [4, 6, 7, 8, 13,21, 1, 1],
                [7, 9, 9, 9, 12,1, 1, 9]])

def decoding():
   
    
    loaded_data = np.load("EncodedFile.npz")
    crEncoded,cbEncoded,(channelLength,sample_rate)=loaded_data["array1"],loaded_data["array2"],loaded_data["array3"]
    channelWidth=channelLength
    invZigzagFirstChannel=np.zeros((channelLength, channelWidth))
    invDCTFirstChannel=np.zeros((channelLength, channelWidth))
    FirstChannel=np.zeros((channelLength, channelWidth))

    invZigzagSecondChannel=np.zeros((channelLength, channelWidth))
    invDCTSecondChannel=np.zeros((channelLength, channelWidth))
    SecondChannel=np.zeros((channelLength, channelWidth))


    crDecoded=run_length_decode_2d(crEncoded,channelLength,channelWidth)
    cbDecoded=run_length_decode_2d(cbEncoded,channelLength,channelWidth)

    print("in decoding before  Normlization and after run length decode  :")
    print(np.max(crDecoded),np.min(crDecoded) )
    print(crDecoded[0:5])


# Extend back to the original range

    hBlocksForC = int(channelLength / windowSize)  # number of blocks in the horizontal direction for Matrix
    vBlocksForC = int(channelWidth / windowSize)  # number of blocks in the vertical direction for Matrix

    for i in range(vBlocksForC):
        for j in range(hBlocksForC):
            invZigzagFirstChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = inverse_zigzag(
                crDecoded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
            invDCTFirstChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = (
                invZigzagFirstChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] * QTC)
            FirstChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.idct(
                invDCTFirstChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
            
            invZigzagSecondChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = inverse_zigzag(
                cbDecoded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
            invDCTSecondChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = (
                invZigzagSecondChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] * QTC)
            SecondChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.idct(
                invDCTSecondChannel[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
    


    FirstChannel=FirstChannel+128 
    SecondChannel=SecondChannel+128
    # Convert flattened arrays to column vectors
    FirstChannel = FirstChannel.reshape(-1, 1)
    SecondChannel = SecondChannel.reshape(-1, 1)

    # Concatenate the vectors along axis 1 to create a matrix
    decoded_signal = np.concatenate((FirstChannel, SecondChannel), axis=1)
    wavfile.write("output_audio.wav", sample_rate, decoded_signal.astype(np.int16))