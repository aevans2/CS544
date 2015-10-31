from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split

# read in data 
datainp = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data', header=None, na_values=['?'])

#clean data by removing rows with missing entries (?)
datainp = datainp[pd.notnull(datainp[6])]

# break data array into train/test sets (75%/25% split)
a = datainp
a_train, a_test = train_test_split(a, test_size=.25, random_state=42)

# assign data set for training
X = a_train.ix[:,1:9]
y = a_train.ix[:,10]

# create SVM
clf = svm.SVC(kernel='linear', gamma=0.0001, C=1000)

# train SVM on the training data
clf.fit(X,y)

# assign data set for testing
testData = a_test.ix[:,1:9]

# predicted values for test set
prediction = clf.predict(testData).tolist()
predictPos = prediction.count(4)
predictNeg = prediction.count(2)

# actual values for test set
actual = (a_test.ix[:,10]).tolist()
actualPos = actual.count(4)
actualNeg = actual.count(2)

print("Actual Positives: " + str(actualPos))
print("Actual Negatives: " + str(actualNeg))

# calculate error metrics
falsePos = abs(actualPos - predictPos)
falseNeg = abs(actualNeg - predictNeg)
totalEx = predictPos + predictNeg

# calculate true positives
truePos = 0
for i, j in zip(actual, prediction):
        if i == j and i == 4:
            truePos += 1

# calculate true negatives
trueNeg = 0
for i, j in zip(actual, prediction):
        if i == j and i == 2:
            trueNeg += 1
        
print("True Positives: " + str(truePos))
print("True Negatives: " + str(trueNeg))

# calculate false positives
falsePos = actualNeg - trueNeg

# calculate false negatives
falseNeg = actualPos - truePos

print("False Positives: " + str(falsePos))
print("False Negatives: " + str(falseNeg))

# calculate prediction accuracy
print("Accuracy: " + format(float((truePos + trueNeg)/float(totalEx)*100), '.2f') + "%")

# calculate prediction precision

print("Precision: " + format((float(truePos / float(truePos + falsePos))*100), '.2f') + "%")

# calculate prediction recall

print("Recall: " + format((float(truePos / float(truePos + falseNeg))*100), '.2f') + "%")