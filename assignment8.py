import numpy as np
import pandas as pd
from scipy import stats
from sklearn import linear_model as lm, metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_predict

# Part 1.1
# Accuracy refers to the closeness of a measured value to a standard or known value meanwhile precision refers to the
# closeness of two or more measurements to each other. Precision is independent of accuracy. You can be very precise but inaccurate, as well as vice versa.
# (reference: https://labwrite.ncsu.edu/Experimental%20Design/accuracyprecision.html)

# Part 1.2
# Precision (also called positive predictive value) is the fraction of relevant instances among the retrieved instances.
# Recall is the fraction of relevant instances that have been retrieved over the total amount of relevant instances.
# (reference https://en.wikipedia.org/wiki/Precision_and_recall)

# Part 1.3
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
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

msk = np.random.rand(len(df)) < 0.8  # splitting the data into test and training set
train = df[msk]
test = df[~msk]


inputset = [inpA, inpB]
overallResults = []

for item in inputset:
    X = df[item].values.reshape(-1, len(item))
    Y = df['Diagnosis']              # comparing against actual diagnose
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
     
    Y_prediction = cross_val_predict(lmodel, X_train, Y_train, cv=10)
    target_names = ['M', 'B']
    print ( classification_report(Y_test,prediction, target_names=target_names) )

    overallResults.append(accuracies)

print('Results based on size', str(np.mean(np.array(overallResults[0]).astype(np.float))))
print('Results based on symmetry', str(np.mean(np.array(overallResults[1]).astype(np.float))))

#Printed result:
# inpA = ['SE area', 'mean perimeter', 'mean radius']  # size based: in this case bigger is not better
#
#             precision    recall  f1-score   support
#
#          M       0.91      0.98      0.94        43
#          B       0.90      0.69      0.78        13
#
#avg / total       0.91      0.91      0.91        56
#
# inpB = ['mean smoothness', 'SE area', 'Worst symmetry']  # symmetry based: less symmetrical, higher possibility of being malign
#
#             precision    recall  f1-score   support
#
#          M       0.91      1.00      0.96        43
#          B       1.00      0.69      0.82        13
#
#avg / total       0.93      0.93      0.92        56
#
#
# Some conclusion to the result.
# First we want to reserve us that we base the classification_report on the last K-fold. 
# We get roughly the same result in all folds.
# For inpB we can see that the precision for is perfect. Our model can predict B with 100% precision.  
# For both of our models and both our classes we see that precision, recall and f1-score is close to 1, which is the highest score.
# This tells us that the model makes prediction with high precision
#



# Part 2: Population and t-test

# Part 2.1
# The t-test is a method used to determine the difference within #two groups. It compares two averages (means) and tells you if #they are different from each other. Every t-value has a p-#value to go with it. A p-value is the probability
# that the #results from your sample data occurred by chance. P-values are #from 0% to 100%. They are usually written  as a decimal. (The #smalles the P-value the better)

# Part 2.2 & 2.3
df = pd.read_csv('brain_size.csv',";",names=["Gender","FSIQ","VIQ","PIQ","Weight","Height","MRI_Count"],header=0)

danishAvgHeight = 71
AMURICA_FUCK_YEAH_AvgHeight = 68.4
df = df[df.Height != "."]

heights = [ float(h) for h in df["Height"] ]
avg  =  sum(heights) / float(len(heights))
print("the average height in the given dataset is :", avg)

dktt = stats.ttest_1samp(heights, danishAvgHeight)
usatt = stats.ttest_1samp(heights, AMURICA_FUCK_YEAH_AvgHeight)

print("The Danish population ttest gave :" , dktt.statistic , " , and p-value: ", dktt.pvalue)
print("The American population ttest gave :" , usatt.statistic , " , and p-value: ", usatt.pvalue)


# According to the T-values and P-values we observe that while using dataset of danish population our hypothesis was not
# confirmed, better said, the result denies it.
# While using American dataset we can see that our probability is on high level (84%) with small deviation of only 0.19,
# according to this we can assume that american dataset is closer to the reality from csv (and that Danes are tall).
