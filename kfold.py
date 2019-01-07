from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.neural_network import MLPClassifier

training = pd.read_csv('training_data.csv', header=None, usecols=list(range(40))).values
labels = pd.read_csv('labels.csv', header=None).values
training, labels = shuffle(training, labels)

sc = StandardScaler()
training = sc.fit_transform(training)
pca = PCA(.8)
training = pca.fit_transform(training)

#Comment the next line (which uses SVM) and uncomment the clf = MLPClassifier() to use the Neural network
#clf = svm.SVC(gamma='scale')
clf = MLPClassifier()
scores = cross_val_score(estimator=clf, X=training, y=labels.ravel(), cv=5, n_jobs=-1)
print(scores)
