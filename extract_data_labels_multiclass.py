
import numpy as np
import os
from preprocessing_functions import *
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1D, GlobalAveragePooling1D, Embedding, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import ModelCheckpoint
import datetime
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


wavs_path_a = './classification-of-heart-sound-recordings-the-physionet-computing-in-cardiology-challenge-2016-1.0.0/training-f/'
all_labels_file = './annotations/annotations/Online Appendix_training set.csv'
#read only the first 4 columns
#rename the header of the csv file names=["filename", "database", "label"]

# df = pd.read_csv(all_labels_file, usecols=[0, 1, 3], names=["filename", "database", "label"])


df = pd.read_csv(all_labels_file, usecols=[0,1,3])
df.columns = ["filename", "database", "label"]
print(df.head())


data_matrix = []
labels = []
sample_length = 6 * 500 #(six seconds of signal with 500Hz sampling rate)

for file in sorted(os.listdir(wavs_path_a)):   
    if file.endswith('.wav'):
        print(file, 'file')
        file_without_extension = os.path.splitext(file)[0]
        
        if df.loc[df['filename'] == file_without_extension].empty:
            continue
        wav, sr = read_wav_scipy_wavfile(wavs_path_a + file)
        resampled_wav = resample_scipy(wav, sr, 500)
        print(resampled_wav.shape, 'resampled_wav shape')
        if resampled_wav.shape[0] < sample_length:
            continue
        no_windows = resampled_wav.shape[0] // sample_length
        for i in range(no_windows):
            data_matrix.append(resampled_wav[i*sample_length:(i+1)*sample_length])
            #If the file name does not exist in the dataframe, skip it
            labels.append(df.loc[df['filename'] == file_without_extension, 'label'].iloc[0])

# print(len(data_matrix), 'data_matrix') 
array_matrix= np.array(data_matrix)
labels = np.array(labels)
print(array_matrix.shape, 'array shape')
print(len(labels), 'labels')     
print(labels, 'labels')

save_path = './dala_labels_multiclass/f/'
np.save(save_path + 'array_matrix_f.npy', array_matrix)
np.save(save_path + 'labels_f.npy', labels)        
# print(wav.shape, 'wav shape')
# print(wav, 'wav')
# print(sr, 'sr')


a = np.load('./dala_labels_multiclass/a/array_matrix_a.npy')
b = np.load('./dala_labels_multiclass/b/array_matrix_b.npy')
c = np.load('./dala_labels_multiclass/c/array_matrix_c.npy')
d = np.load('./dala_labels_multiclass/d/array_matrix_d.npy')
e = np.load('./dala_labels_multiclass/e/array_matrix_e.npy')
# f = np.load('./training_matrixes&labels/f/array_matrix_f.npy')

# print(a.shape, 'a shape')
# print(b.shape, 'b shape')
# print(c.shape, 'c shape')
# print(d.shape, 'd shape')
# print(e.shape, 'e shape')
# print(f.shape, 'f shape')



a_labels = np.load('./dala_labels_multiclass/a/labels_a.npy')
b_labels = np.load('./dala_labels_multiclass/b/labels_b.npy')
c_labels = np.load('./dala_labels_multiclass/c/labels_c.npy')
d_labels = np.load('./dala_labels_multiclass/d/labels_d.npy')
e_labels = np.load('./training_matrixes&labels/e/labels_e.npy')
# f_labels = np.load('./training_matrixes&labels/f/labels_f.npy')

