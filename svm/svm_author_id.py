#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append(r"D:\machine_learning\ud120-projects\tools")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


#########################################################
### your code goes here ###
# SVM classifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
# Create classifier
clf = SVC(kernel='rbf', C= 1.0)

# Train
t0 = time()
clf.fit(features_train, labels_train)
print("Training time:", round(time()-t0, 3), "s")

# Predict
t1 = time()
pred = clf.predict(features_test)
print("Predicting time:", round(time()-t1, 3), "s")

# Calculate accuracy
acc = accuracy_score(labels_test, pred)
print("Accuracy:", acc)
#########################################################

#########################################################
'''
You'll be Provided similar code in the Quiz
But the Code provided in Quiz has an Indexing issue
The Code Below solves that issue, So use this one
'''

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]

#########################################################
