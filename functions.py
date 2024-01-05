from collections import Counter

import numpy as np
from collections import Counter

import pickle



import numpy as np
from scipy.io import wavfile

def read_sound_file(input_file):
    # Example usage:
    sample_rate, audio_data = wavfile.read(input_file)
    return sample_rate ,audio_data


def padding_channels(first_channel,second_channel):

    while np.sqrt(first_channel.size)-int(np.sqrt(first_channel.size))!=0:
        first_channel=np.append(first_channel,0)

    while np.sqrt(second_channel.size)-int(np.sqrt(second_channel.size))!=0:
        second_channel=np.append(second_channel,0)
    
    return first_channel,second_channel


def convert_channels_2d(first_channel,second_channel):
    matrix_size = int(np.sqrt(len(second_channel)))
    first_channel_2d = np.reshape(first_channel, (matrix_size, matrix_size))
    second_channel_2d = np.reshape(second_channel, (matrix_size, matrix_size))
    return first_channel_2d,second_channel_2d
 
 
def convert_audio_to_2d(file_name):
    sample_rate,audio_data=read_sound_file(file_name)
    first_channel =audio_data[:,0]
    second_channel =audio_data[:,1]
    first_channel,second_channel=padding_channels(first_channel,second_channel)
    
    return sample_rate,convert_channels_2d(first_channel,second_channel)




def zigzag(matrix: np.ndarray) -> np.ndarray:
    """
    computes the zigzag of a quantized block
    :param numpy.ndarray matrix: quantized matrix
    :returns: zigzag vectors in an array
    """
    # initializing the variables
    h = 0
    v = 0
    v_min = 0
    h_min = 0
    v_max = matrix.shape[0]
    h_max = matrix.shape[1]
    i = 0
    output = np.zeros((v_max * h_max))
    x=0
    y=0 
    dict={}
    while (v < v_max) and (h < h_max):
        
        if ((h + v) % 2) == 0:  # going up
            if v == v_min:
                output[i] = matrix[v, h]  # first line  
                
                if h == h_max:
                    v = v + 1
                else:
                    h = h + 1
                i = i + 1
            elif (h == h_max - 1) and (v < v_max):  # last column
                output[i] = matrix[v, h]
                
                
                v = v + 1
                i = i + 1
            elif (v > v_min) and (h < h_max - 1):  # all other cases
                output[i] = matrix[v, h]
                
                v = v - 1
                h = h + 1
                i = i + 1
        else:  # going down
            if (v == v_max - 1) and (h <= h_max - 1):  # last line
                output[i] = matrix[v, h]
                
                h = h + 1
                i = i + 1
            elif h == h_min:  # first column
                output[i] = matrix[v, h]
                

                if v == v_max - 1:
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif (v < v_max - 1) and (h > h_min):  # all other cases
                output[i] = matrix[v, h]
                
                v = v + 1
                h = h - 1
                i = i + 1
        if (v == v_max - 1) and (h == h_max - 1):  # bottom right element
            output[i] = matrix[v, h]
            
            break
    return np.reshape(output, (8, 8))

def inverse_zigzag(matrix: np.ndarray) -> np.ndarray:
    """
    computes the zigzag of a quantized block
    :param numpy.ndarray matrix: quantized matrix
    :returns: zigzag vectors in an array
    """
    # initializing the variables

    v_max = matrix.shape[0]
    h_max = matrix.shape[1]
    output = np.zeros((v_max ,h_max))
    dict={}
    with open('zigzag_mapping.pkl', 'rb') as pickle_file:
        dict = pickle.load(pickle_file)
    for i in range(v_max):
        for j in range(h_max):
            output[i,j]=matrix[dict[i,j][0],dict[i,j][1]]

    return output
def trim(array: np.ndarray) -> np.ndarray:
    """
    in case the trim_zeros function returns an empty array, add a zero to the array to use as the DC component
    :param numpy.ndarray array: array to be trimmed
    :return numpy.ndarray:
    """
    trimmed = np.trim_zeros(array, 'b')
    if len(trimmed) == 0:
        trimmed = np.zeros(1)
    return trimmed


def run_length_encoding(array: np.ndarray) -> list:
    """
    finds the intermediary stream representing the zigzags
    format for DC components is <size><amplitude>
    format for AC components is <run_length, size> <Amplitude of non-zero>
    :param numpy.ndarray array: zigzag vectors in array
    :returns: run length encoded values as an array of tuples
    """
    encoded = list()
    run_length = 0
    eob = ("EOB",)

    for i in range(len(array)):
        for j in range(len(array[i])):
            trimmed = trim(array[i])
            if j == len(trimmed):
                encoded.append(eob)  # EOB
                break
            if i == 0 and j == 0:  # for the first DC component
                encoded.append((int(trimmed[j]).bit_length(), trimmed[j]))
            elif j == 0:  # to compute the difference between DC components
                diff = int(array[i][j] - array[i - 1][j])
                if diff != 0:
                    encoded.append((diff.bit_length(), diff))
                else:
                    encoded.append((1, diff))
                run_length = 0
            elif trimmed[j] == 0:  # increment run_length by one in case of a zero
                run_length += 1
            else:  # intermediary steam representation of the AC components
                encoded.append((run_length, int(trimmed[j]).bit_length(), trimmed[j]))
                run_length = 0
            # send EOB
        if not (encoded[len(encoded) - 1] == eob):
            encoded.append(eob)
    return encoded


def get_freq_dict(array: list) -> dict:
    """
    returns a dict where the keys are the values of the array, and the values are their frequencies
    :param numpy.ndarray array: intermediary stream as array
    :return: frequency table
    """
    #
    data = Counter(array)    
    print(data.items())
    result = {k: d / len(array) for k, d in data.items()}
    return result


def find_huffman(p: dict) -> dict:
    """
    returns a Huffman code for an ensemble with distribution p
    :param dict p: frequency table
    :returns: huffman code for each symbol
    """
    if len(p) == 1:
        return {list(p.keys())[0]: ''}
    # Base case of only two symbols, assign 0 or 1 arbitrarily; frequency does not matter
    if len(p) == 2:
        return dict(zip(p.keys(), ['0', '1']))
    
    # print(len(p))
    # Create a new distribution by merging lowest probable pair
    p_prime = p.copy()
    a1, a2 = lowest_prob_pair(p)
    p1, p2 = p_prime.pop(a1), p_prime.pop(a2)
    p_prime[a1 + a2] = p1 + p2

    # Recurse and construct code on new distribution
    c = find_huffman(p_prime)
    ca1a2 = c.pop(a1 + a2)
    c[a1], c[a2] = ca1a2 + '0', ca1a2 + '1'

    return c


def lowest_prob_pair(p):
    # Return pair of symbols from distribution p with lowest probabilities
    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]


def run_length_encode_2d(array):
    flattenArray=array
    encoded_array = []
    cnt=0
    run=0 
    num=flattenArray[0]
    for NextNum in flattenArray[0:] :
        if(NextNum==num):
            cnt+=1 
        else:
            encoded_array.append((num, cnt))
            run+=cnt
            cnt=1
            num=NextNum
            
        # Append the last run
    encoded_array.append((num, cnt))
    return encoded_array

def run_length_decode_2d(encoded_array,wid,MatLen):
    decoded_rows = []
    cnt=0 
    for run in encoded_array:
        value, length = run
        cnt+=length
        decoded_rows.extend([value] * length)
    return np.array(decoded_rows).reshape(wid, MatLen)

def normalize(value, min_orig, max_orig, min_target, max_target):
    normalized = min_target + (value - min_orig) * (max_target - min_target) / (max_orig - min_orig)
    return normalized

def restore(normalized, min_orig, max_orig, min_target, max_target):
    
    restored = min_orig + (normalized - min_target) * (max_orig - min_orig) / (max_target - min_target)

    return restored

def inverse_huffman(huffman_code):
    inverse_dict = {}
    for symbol, code in huffman_code.items():
        inverse_dict[code] = symbol
    return inverse_dict


def inverse_get_freq_dict(freq_dict):
    array = []
    for bits, (value,frequency) in freq_dict.items():
        count = int(frequency * len(array))
        array.extend([value] * count)
    return array