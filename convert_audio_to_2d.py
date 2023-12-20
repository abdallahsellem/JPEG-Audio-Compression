import numpy as np
from scipy.io import wavfile

def read_sound_file(input_file):
    # Example usage:
    input_file = 'file_example_WAV_5MG.wav'
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
