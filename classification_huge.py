from pandas import read_csv
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.neural_network import MLPClassifier

# Classifies and finds the accuracy of the features extracted from the 200 ms frames
# of the file with speech and music segments using the 'huge_file.txt'
# that was created by the join_wavs() function of feature_extraction.py

# Read the datasets
training = read_csv('training_data.csv', header=None).values
labels = read_csv('labels.csv', header=None).values
huge_file = read_csv('datasets/speech_and_music_huge_200ms_frames.csv', header=None).values

training, labels = shuffle(training, labels)
sc = StandardScaler()
training = sc.fit_transform(training)
pca = PCA(.8)
training = pca.fit_transform(training)
# Comment the next line (which uses SVM) and uncomment the clf = MLPClassifier() to use the Neural network
clf = svm.SVC(gamma='scale')
#clf = MLPClassifier()
clf.fit(training, labels.ravel())

huge_file = sc.transform(huge_file)
huge_file = pca.transform(huge_file)
huge_results = clf.predict(huge_file).astype(int)
final = np.bincount(huge_results)
speech_perc = final[1]*100/sum(final)

labels = []
with open('datasets/huge_file.txt') as f: # Change the path if needed
    mod=200
    boo = False
    for line in f:
        kind, start, stop = line.split()
        start, stop = int(start), int(stop)
        start += 200-mod
        frames = (stop-start)/200
        frames_int = int(frames)
        if kind == 'genres,': label = 0
        else: label = 1
        if boo: frames_int+=1
        labels.extend(frames_int*[label])
        mod = stop%200
        if mod > 100:
            labels.append(label)
            boo = False
        elif mod == 0:
            mod = 200
            boo = False
        else:
            boo = True
    if mod < 100:
        labels.append(label)
        
labels = np.array(labels)
huge_results_copy = huge_results
for i in range(len(huge_results) - 10):
    if huge_results[i] == huge_results[i + 9] == 0:
        for j in range(i+1, i+9):
            huge_results[j] = 0
    elif huge_results[i] == huge_results[i+4] == 1:
        for j in range(i+1, i+4):
            huge_results[i+1] = 1
accuracy = np.sum(labels == huge_results)/len(labels)
print('Accuracy:', accuracy)
