# python 3.9.5
from math import ceil
import cv2


import numpy as np
import pickle
from functions import *
from convert_audio_to_2d import *
# define quantization tables
windowSize=8
def encoding():
    QTY = np.array([[16, 11, 10, 16, 24, 40, 51, 61],  #  quantization table
                    [12, 12, 14, 19, 26, 48, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])

    QTC = np.array([[99, 99, 99, 99, 99, 99, 99, 99],  #  quantization table
                    [99, 99, 26, 66, 99, 99, 99, 99],
                    [99, 99, 56, 99, 99, 99, 99, 99],
                    [99, 66, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99],
                    [99, 99, 99, 99, 99, 99, 99, 99]])


    #read audioFile 
    sample_rate,(first_channel,second_channel) = convert_audio_to_2d("file_example_WAV_5MG.wav")
    width = len(first_channel[0])
    height = len(first_channel)
    first_channel = np.zeros((height, width), np.float16) + first_channel
    second_channel = np.zeros((height, width), np.float16) + second_channel 
    # size of the image in bits before compression
    totalNumberOfBitsWithoutCompression = len(first_channel) * len(first_channel[0]) * 8 + len(second_channel) * len(second_channel[0]) * 8 
    print(totalNumberOfBitsWithoutCompression)

    # check if padding is needed,
    # if yes define empty arrays to pad each channel DCT with zeros if necessary
    channelWidth, channelLength = ceil(len(first_channel[0]) / windowSize) * windowSize, ceil(len(first_channel) / windowSize) * windowSize
    if (len(first_channel[0]) % windowSize == 0) and (len(first_channel) % windowSize == 0):
        first_channelPadded = first_channel.copy()
        second_channelPadded = second_channel.copy()

    else:
        first_channelPadded = np.zeros((channelLength, channelWidth))
        second_channelPadded= np.zeros((channelLength, channelWidth))
        for i in range(len(first_channel)):
            for j in range(len(first_channel[0])):
                first_channelPadded[i, j] += first_channel[i, j]
                second_channelPadded[i, j] += second_channel[i, j]


    # # get DCT of each channel
    # # define three empty matrices
    crDct, cbDct =  np.zeros((channelLength, channelWidth)), np.zeros((channelLength, channelWidth))

    # number of iteration on x axis and y axis to calculate channels cosine transforms values
    hBlocksForC = int(len(crDct[0]) / windowSize)  # number of blocks in the horizontal direction for chrominance
    vBlocksForC = int(len(crDct) / windowSize)  # number of blocks in the vertical direction for chrominance

    # # define 3 empty matrices to store the quantized values
    crq, cbq = np.zeros((channelLength, channelWidth)), np.zeros((channelLength, channelWidth))
    crZigzag = np.zeros((channelLength,channelWidth))
    cbZigzag = np.zeros((channelLength,channelWidth))


    # either crq or cbq can be used to compute the number of blocks
    cCounter = 0
    x=None
    x2=None
    for i in range(vBlocksForC):
        for j in range(hBlocksForC):
            crDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.dct(
                first_channelPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
            crq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = np.floor(
                crDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] / QTY)
            crZigzag[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = zigzag(
                crq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
            
            cbDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = cv2.dct(
                second_channelPadded[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])
            cbq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] = np.floor(
                cbDct[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize] / QTY)
            cbZigzag[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize]= zigzag(
                cbq[i * windowSize: i * windowSize + windowSize, j * windowSize: j * windowSize + windowSize])

    crZigzag = crZigzag.astype(np.int16)
    cbZigzag = cbZigzag.astype(np.int16)

    # find the run length encoding for each channel
    # then get the frequency of each component in order to form a Huffman dictionary

    crEncoded = run_length_encode_2d(crZigzag)

    cbEncoded = run_length_encode_2d(cbZigzag)
    
    dt = np.dtype([('0', np.int8), ('1', np.int8)])
    pairs_array1=np.array(crEncoded,dt)
    pairs_array2=np.array(cbEncoded,dt)
    np.save('pairs_file1.npy', pairs_array1)
    np.save('pairs_file2.npy', pairs_array2)
    

    

