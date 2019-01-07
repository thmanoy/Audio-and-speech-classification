# Change this to the folder where the csv files are
csv_folder = '/home/user/Desktop/ΗΜΜΥ/Τεχνολογία του ήχου και της εικόνας/Εργασία/datasets/genres_200ms_frames_csv_files/'

import pandas as pd
from numpy import bincount
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
import os

# Classifies the features that were exported using the
# function extract_testing_files() of feature_extraction.py

training = pd.read_csv('training_data.csv', header=None).values
labels = pd.read_csv('labels.csv', header=None).values

training, labels = shuffle(training, labels)
sc = StandardScaler()
training = sc.fit_transform(training)
pca = PCA(.8)
training = pca.fit_transform(training)
# Comment the next line (which uses SVM) and uncomment the clf = MLPClassifier() to use the Neural network
clf = svm.SVC(gamma='scale')
#clf = MLPClassifier()
clf.fit(training, labels.ravel())
results = []
os.chdir(csv_folder)
csvs = os.listdir()
num_csv = len(csvs)
i=0
errors=0

for csv in csvs:
    testing = pd.read_csv(csv, header=None).values
    testing = sc.transform(testing)
    try:
        testing = pca.transform(testing)
        pred = clf.predict(testing).astype(int)
    except ValueError:
        errors+=1
        print('Value Errors:', errors)
        continue
    fin = bincount(pred)
    if len(fin) == 1:
        if pred[0] == 0: results.append(0)
        else: results.append(1)
    else:
        if fin[0] > fin[1]: results.append(0)
        else: results.append(1)
    i+=1
    print(i/num_csv*100, '% completed')
    
final = bincount(results)
music_perc = 100*final[0]/sum(final)
speech_perc = 100*final[1]/sum(final)
