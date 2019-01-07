# Change this to the folder of the datasets
training_path = '/home/user/Desktop/ΗΜΜΥ/Τεχνολογία του ήχου και της εικόνας/Εργασία/datasets'

import librosa
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy as np
import os
import soundfile as sf
from random import shuffle
from sys import getsizeof

# Function that joins randomly selected sound files from the folders in the list foldername
# (with music and speech files) into the file 'speech_and_music_huge.wav'
def join_wavs(foldername=['genres', 'speech']):
    os.chdir(training_path)
    f=[]
    for fn in foldername:
        for root, dirs, files in os.walk(fn):
            files = [root + '/' + s for s in files]
            map(os.path.abspath, files)
            f.extend(files)
            print(files)
    sounds = []
    shuffle(f)
    i=0
    for sound in f:
        sounds.append(AudioSegment.from_file(sound))
        i+=1
        if getsizeof(sounds) > 5000: break # It uses too much RAM without this
        print(getsizeof(sounds))
    # Creating file with the start and ending of each segment of speech and music
    with open('huge_file.txt', 'w') as fi:
        j=0
        su=0
        for string in f:
            if j<i:
                string = string[:6]
                try:
                    fi.write('%s, %d %d\n' % (string, su, su+len(sounds[j])))
                except IndexError:
                    print(i, j)
                    exit(1)
                su+=len(sounds[j])
                j+=1
            else: break
    all_together = AudioSegment.empty()
    for sound in sounds:
        all_together += sound
    all_together.export('speech_and_music_huge.wav', format='wav')

# Splits the files of the folder with name 'foldername', along with its subfolders, into frames of 200 ms
# and makes a folder for each original file to put its frames into
def split_to_frames(foldername):
    os.chdir(training_path)
    newdir = foldername + '_200ms_frames'
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    f = []
    for root, dirs, files in os.walk(foldername):
        files = [root + '/' + s for s in files]
        map(os.path.abspath, files)
        print(files)
        f.extend(files)
    j = 0
    for wav in f:
        j=j+1
        os.makedirs(newdir + '/' + str(j))
        myaudio = AudioSegment.from_file(wav)
        chunk_length_ms = 200 # 200 ms chunk of audio
        chunks = make_chunks(myaudio, chunk_length_ms)
        for i, chunk in enumerate(chunks):
            chunk_name = str(j) + '_' + "{0}.wav".format(i)
            if j % 10 == 0 and i==0: print ("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")
            os.rename(chunk_name, newdir + '/' + str(j) + '/' + chunk_name)

def split_to_frames_training(foldername):
    cwd = os.getcwd() + '/'
    newdir = foldername + '_200ms_frames'
    if not os.path.exists(newdir):
        os.makedirs(newdir)
    os.chdir(foldername)
    f = []
    for root, dirs, files in os.walk(foldername):
        files = [root + '/' + s for s in files]
        map(os.path.abspath, files)
        print(files)
        f.extend(files)
    j=0
    for wav in f:
        myaudio = AudioSegment.from_file(wav)
        chunk_length_ms = 200 # 200 ms chunk of audio
        chunks = make_chunks(myaudio, chunk_length_ms)
        for i, chunk in enumerate(chunks):
            j+=1
            chunk_name = "{0}.wav".format(j)
            if j % 10 == 0: print ("exporting", chunk_name)
            chunk.export(chunk_name, format="wav")
            os.rename(chunk_name, cwd + newdir + '/' + chunk_name)

# Splits a single wav file to files of 200 ms
def split_file_to_frames(filename):
    wav = AudioSegment.from_file('speech_and_music_huge.wav')
    chunks = make_chunks(wav, 200) # 200 ms chunk of audio
    os.makedirs('speech_and_music_huge_200ms_frames')
    os.chdir('speech_and_music_huge_200ms_frames')
    for i, chunk in enumerate(chunks):
        	chunk_name = '{0}.wav'.format(i)
        	chunk.export(chunk_name, format='wav')
	
# Used for sorting a list of files later
def cnt(e):
    return int(e[:-4])

# Performs feature extraction on the files that are in the folder 'foldername' without testing for subfolders
def extract_folder(foldername):
    os.chdir(training_path + '/' + foldername)
    wavs = os.listdir()
    wavs.sort(key=cnt)
    files = len(wavs)
    errors, i = 0, 0
    for wav in wavs:
        try:
            temp = np.hstack(extract_feature(wav))
        except:
            errors += 1
            print('Errors: ', errors)
            continue
        if 'fets' not in locals():
            fets = temp.size
            features = np.empty([files, fets])
            print(fets)
#            files = 1800
        features[i, :] = temp
        i+=1
#        if i==1800: break
        perc = i/files*100
        if perc % 1 < 0.05: print(i, ':', foldername + ' ' + str("{0:.2f}".format(perc)) + ' % completed')
    os.chdir(training_path)
    np.savetxt(foldername + '_200ms_frames.csv', features, delimiter=', ')

# Performs feature extraction on each folder of the folder 'foldername'
# and for each of its subfolders, it creates a different csv file
# representing the original audio file
def extract_testing_files(foldername):
    os.chdir(training_path)
    csvf = foldername + '_csv_files'
    if not os.path.exists(csvf): os.makedirs(csvf)
    files = len(os.listdir(training_path + '/' + foldername))
    print(files)
    foldername2 = '/' + foldername + '/'
    errors, j = 0, 0
    while os.path.exists(training_path + foldername2 + str(j+1)):
        i=0
        j=j+1
        os.chdir(training_path + foldername2 + str(j))
        rows = len(os.listdir())
        f = os.listdir()
        boole = True
        for wav in f:
            try:
                temp = np.hstack(extract_feature(wav))
            except:
                errors += 1
                print('Errors: ', errors)
                continue
            if 'fets' not in locals():
                fets = temp.size
                print(fets)
            if boole:
                features = np.empty([rows, fets])
                boole = False
            features[i, :] = temp
            i=i+1
        perc = j/files*100
        if perc % 1 < 0.05: print(j, ':', foldername + ' ' + str("{0:.2f}".format(perc)) + ' % completed')
            
        os.chdir(training_path + '/' + csvf)
        np.savetxt('{0}.csv'.format(j), features, delimiter=', ')

def extract_feature(file_name):
    """Generates feature input (mfccs, chroma, mel, contrast, tonnetz).
    -*- author: mtobeiyf https://github.com/mtobeiyf/audio-classification -*-
    """
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T

    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

if __name__ == '__main__':
    # Split and extract training files
    split_to_frames_training('music_training')
    split_to_frames_training('speech_training')
    extract_folder('music_training_200ms_frames')
    extract_folder('speech_training_200ms_frames')
    # Split and extract testing files
    split_to_frames('genres')
    split_to_frames('speech')
    extract_testing_files('genres_200ms_frames')
    extract_testing_files('speech_200ms_frames')
    # Create file from the datasets
    join_wavs()
    # Split and feature extract the file
    split_file_to_frames('speech_and_music_huge.wav')
    extract_folder('speech_and_music_huge_200ms_frames')
