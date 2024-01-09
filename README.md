# JPEG-Audio-Compressor
A Python program that compresses Audio based on the JPEG compression algorithm.

This program takes as input a Audio File (eg: .wav).

The File is read using the wavfile from scipy library , then we are adding padding to the file and converting the 2 channel in sterio audio to matrix so we can apply JPEG.

Each channel is divided into 8 × 8 blocks – and is padded with zeros if needed. Each block undergoes a discrete cosine transform, where in the resulting block, the first component of each block is called the DC coefficient, and the other 63 are AC components.

DC coefficients are encoded using DPCM as follows: \<size in bits\>, \<amplitude\>. AC components are encoded using run length in the following way: \<run length, size in bits\>, \<amplitude\>, while using zigzag scan on the block to produce longer runs of zeros.
  
An intermediary stream consists of encoded DC and AC components, and an EOB (end of block) to mark the end of the block. To achieve a higher compression rate, all zero AC components are trimmed from the end of the zigzag scan.
  
Finally , we are storing the Encoded binary code using numpy 


### to run code 

- write in terminal "python main.py <file_name> <Operation_type["encode","decode"]>"

***Examples:***

for encoding

    python main.py file_example.wav encode

for decoding 

    python main.py EncodedFile.bin decode


