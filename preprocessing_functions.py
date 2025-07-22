import os
import scipy.io
import pandas as pd

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample as scipy_resample

def PCG_segmentation(directory, filename):
    df = pd.DataFrame(columns=("S1", "systole", "S2", "diastole"))
    file_path = os.path.join(os.path.curdir, "data", directory, filename)
    annotations_path = os.path.join("annotations\\hand_corrected", 
                                    directory+"_StateAns", 
                                    filename.replace(".wav", "_StateAns.mat"))
    signal = wavfile.read(file_path)[1]
    annotations =  scipy.io.loadmat(annotations_path)['state_ans']
    i = 0
    while (annotations[i][1]!= "S1" and i < annotations.shape[0]):
        i+=1
        
    d = {}
    while (i < annotations.shape[0]-1):
        d[annotations[i][1][0][0][0]] = signal[annotations[i][0][0][0]:annotations[i+1][0][0][0]]
        if annotations[i][1][0][0][0] == 'diastole':
            df = df.append(d, ignore_index=True)
            d = {}
        i+=1
    return df


    
def read_wav_scipy_wavfile(audio, dtype='float32'):
    """Loading audio and sample rate from a wav file, using wavfile.read. 
    Parameters
    ----------
    audio               : float32
                          multichannel/mono audio
    dtype               : deafult to float32                      
                          dtype for the resulted wav  
    Returns
    -------
    loaded_audio, sr    : tuple[float32, int]
                          audio with shape [channels, samples] and samplerate
    
    Observation! If the audio was written with a function, use the same function for load.
    """
    sr, loaded_audio = wavfile.read(audio)
    loaded_audio = loaded_audio.astype(dtype)
    return loaded_audio.T, sr

def resample_scipy(wav, old_sr, new_sr):
    """Load and resample audio using scipy.io wavfile and scipy.signal resample
    Parameters
    ----------
    wav                         : float32
                                multichannel/mono audio    
        
    old_sr                      : int
                                original samplerate                            
        
    new_sr                      : int                      
                                desired samplerate  
    Returns
    -------
    resampled_wav               : float32
                                 resampled audio with shape [channels, samples]
    
    Observation! If the audio was written with a function, use the same function for load.
    """
    new_no_samples = round(wav.shape[0] * new_sr / old_sr)
    resampled_wav = scipy_resample(wav.T, new_no_samples).T
    return resampled_wav


def normalize(audio):
     """Normalizes the audio, meaning the resulted samples range between -1 and 1.
     Parameters
     ----------
     audio               : float32
                           multichannel/stereo audio
     Returns
     -------
     normalized_audio    : float32
                           normalized multichannel/stereo audio
     """
     normalized_audio = audio/np.max(np.abs(audio))
     return normalized_audio