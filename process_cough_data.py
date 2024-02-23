import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import sys
import IPython.display as ipd
from enum import Enum
sys.path.append(os.path.abspath('./src'))
from helpers import *
from dataset_gen import *

# Seed the random number generator for augmentation purposes
seed = 1
np.random.seed(seed)

# control params
window_len=0.7
aug_factor=2
# this is the number of samples in a window per fft
n_fft = 2048
# The amount of samples we are shifting after each fft
hop_length = 128
save_path = './data/EdgeAICough/'


data_folder = './public_dataset/train/' #Location of public dataset in your file directory

subj_ids = os.listdir(data_folder)
print("There are {0} subjects".format(len(subj_ids)))

audio_data_all = np.zeros((0, round(window_len*FS_AUDIO), 2))
labels_all = np.zeros(0)
for subj_id in subj_ids:
    audio_data, imu_data, labels, total_coughs = get_samples_for_subject(data_folder, subj_id=subj_id, window_len=window_len, aug_factor=aug_factor)
    audio_data_all = np.concatenate((audio_data_all, audio_data), axis=0)
    labels_all = np.concatenate((labels_all, labels), axis=0)
    print("Audio data shape: {0}".format(audio_data.shape))
    print("Labels shape: {0}".format(labels.shape))
    print("Number of total coughs before augmentation: {0}".format(total_coughs))

data_folder = './public_dataset/test/' #Location of public dataset in your file directory

subj_ids = os.listdir(data_folder)
print("There are {0} subjects".format(len(subj_ids)))

audio_data_test = np.zeros((0, round(window_len*FS_AUDIO), 2))
labels_test = np.zeros(0)
for subj_id in subj_ids:
    audio_data, imu_data, labels, total_coughs = get_samples_for_subject(data_folder, subj_id=subj_id, window_len=window_len, aug_factor=aug_factor)
    audio_data_test = np.concatenate((audio_data_test, audio_data), axis=0)
    labels_test = np.concatenate((labels_test, labels), axis=0)
    print("Audio data shape: {0}".format(audio_data.shape))
    print("Labels shape: {0}".format(labels.shape))
    print("Number of total coughs before augmentation: {0}".format(total_coughs))

import librosa, librosa.display
mel_data_train = np.zeros((audio_data_all.shape[0], 128, 88, 2))
for i,audio_data_train in enumerate(audio_data_all):
    # MEL Short-time Fourier Transformation on our audio data
    # Short-time Fourier Transformation on our audio data
    #mel_signal = librosa.core.stft(audio_data_train[:,0], hop_length=hop_length, n_fft=n_fft)
    mel_signal = librosa.feature.melspectrogram(y=audio_data_train[:,0], sr=FS_AUDIO, hop_length=hop_length, n_fft=n_fft)
    # gathering the absolute values for all values in our audio_stft
    spectrogram = np.abs(mel_signal)
    # Converting the power to decibels
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    mel_data_train[i,:,:,0] = power_to_db

    mel_signal = librosa.feature.melspectrogram(y=audio_data_train[:,1], sr=FS_AUDIO, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    mel_data_train[i,:,:,1] = power_to_db

import librosa, librosa.display
mel_data_test = np.zeros((audio_data_test.shape[0], 128, 88, 2))
for i,audio_data_tst in enumerate(audio_data_test):
    # MEL Short-time Fourier Transformation on our audio data
    # Short-time Fourier Transformation on our audio data
    #mel_signal = librosa.core.stft(audio_data_tst[:,0], hop_length=hop_length, n_fft=n_fft)
    mel_signal = librosa.feature.melspectrogram(y=audio_data_tst[:,0], sr=FS_AUDIO, hop_length=hop_length, n_fft=n_fft)
    # gathering the absolute values for all values in our audio_stft
    spectrogram = np.abs(mel_signal)
    # Converting the power to decibels
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    mel_data_test[i,:,:,0] = power_to_db

    mel_signal = librosa.feature.melspectrogram(y=audio_data_train[:,1], sr=FS_AUDIO, hop_length=hop_length, n_fft=n_fft)
    spectrogram = np.abs(mel_signal)
    power_to_db = librosa.power_to_db(spectrogram, ref=np.max)
    mel_data_test[i,:,:,1] = power_to_db

import torch

if not os.path.exists(save_path):
    os.makedirs(save_path, exist_ok=True)

tensor_data = torch.tensor(mel_data_train)
file_path = "train.pt"
torch.save(tensor_data, save_path+file_path)

tensor_data = torch.tensor(mel_data_test)
file_path = "test.pt"
torch.save(tensor_data, save_path+file_path)

tensor_data = torch.tensor(labels_all)
file_path = "train_labels.pt"
torch.save(tensor_data, save_path+file_path)

tensor_data = torch.tensor(labels_test)
file_path = "test_labels.pt"
torch.save(tensor_data, save_path+file_path)