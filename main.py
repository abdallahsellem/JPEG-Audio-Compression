# python 3.9.5
import argparse
from math import ceil
import cv2


import numpy as np

from functions import *
from encoding import *
from decoding import *

# Create ArgumentParser object
parser = argparse.ArgumentParser(description= 'Simple JPEG Compression for Audio Files \
                                 \n')

# Add command-line arguments
parser.add_argument('filename', type=str, help='Input file to scan')
parser.add_argument('type_of_decoding', type=str, help='Input file to scan')



# Parse the command-line arguments
args = parser.parse_args()
print(args.type_of_decoding)
if(args.type_of_decoding)=="encode" :
    encoding(args.filename)
elif(args.type_of_decoding)=="decode" :
    decoding(args.filename)
else :
    raise(Exception("Wrong Argument ,Please try Again"))
