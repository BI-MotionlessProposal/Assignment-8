# Part 1: Accuracy and recall

# Part 1.1 
# Accuracy refers to the closeness of a measured value to a standard or known value. 
# Precision refers to the closeness of two or more measurements to each other. 
# Precision is independent of accuracy. You can be very precise but inaccurate, as described above. You can also be accurate but imprecise. (reference: https://labwrite.ncsu.edu/Experimental%20Design/accuracyprecision.htm)

# Part 1.2
# Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances.
# Recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances. (reference https://en.wikipedia.org/wiki/Precision_and_recall)

# Part 1.3

import pandas as pd
import numpy as np
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model as lm, metrics
from pandas import read_csv
from sklearn.model_selection import KFold, cross_val_predict
# Assignment 8
from sklearn.metrics import classification_report 


url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
#import urllib.request
# Download the file from `url` and save it locally under `file_name`:
#urllib.request.urlretrieve(url, 'bc.csv')

df = pd.read_csv(url)
df.columns = ['ID number', 'Diagnosis', 'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
              'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension',
              'SE radius', 'SE texture', 'SE perimeter', 'SE area', 'SE smoothness', 'SE compactness', 'SE concavity',
              'SE concave points', 'SE symmetry', 'SE fractal dimension', 'Worst radius', 'Worst texture',
              'Worst perimeter', 'Worst area', 'Worst smoothness', 'Worst compactness', 'Worst concavity',
              'Worst concave points', 'Worst symmetry', 'Worst fractal dimension']
print('Length: ', str(len(df)))

inpA = ['SE area', 'mean perimeter', 'mean radius']  # size based: in this case bigger is not better
inpB = ['mean smoothness', 'SE area', 'Worst symmetry']  # symmetry based: less symmetrical, higher possibility of being malign

msk = np.random.rand(len(df)) < 0.8 # splitting the data into test and training set
train = df[msk]
test = df[~msk]


inputset = [inpA, inpB]
overallResults = []

for item in inputset:
    X = df[item].values.reshape(-1, len(item))
    Y = df['Diagnosis']     # comparing against actual diagnose

    folds = KFold(n_splits=10)

    accuracies = []


    for train, test in folds.split(X, Y):
        X_train, X_test = X[train], X[test]
        Y_train, Y_test = Y[train], Y[test]

        lmodel = lm.LogisticRegression()
        lmodel.fit(X_train, Y_train)

        prediction = cross_val_predict(lmodel, X_test, Y_test, cv=10)

        accuracy = metrics.accuracy_score(Y_test, prediction, normalize=True)

        accuracies.append(accuracy)
     
    # Assignment 8
    Y_prediction = cross_val_predict(lmodel, X_train, Y_train, cv=10)
    target_names = ['M', 'B']
    # Parameters: y_true, y_pred
    print ( classification_report(Y_test,prediction, target_names=target_names) )
    # END Assignment 8

    overallResults.append(accuracies)

print('Results based on size', str(np.mean(np.array(overallResults[0]).astype(np.float))))
print('Results based on symmetry', str(np.mean(np.array(overallResults[1]).astype(np.float))))


# Part 2: Population and t-test

# Part 2.1


# Part 2.2


# Part 2.3


# Part 3 (OPTIONAL): Training a perceptron network


# Part 3.1


# Part 3.2


