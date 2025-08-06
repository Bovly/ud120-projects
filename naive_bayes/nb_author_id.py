#!/usr/bin/python3

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
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


##############################################################
# Enter Your Code Here
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create a Naive Bayes classifier instance
clf = GaussianNB()

# Measure the time to train the classifier
t0 = time()
clf.fit(features_train, labels_train)
print("Training Time:", round(time()-t0, 3), "s")

#Measure the time to predict the labels
t0 = time()
pred = clf.predict(features_test)
print("Predicting Time:", round(time()-t0, 3), "s")

# Calculate the accuracy of the model
accuracy = accuracy_score(labels_test, pred)
print("Accuracy:", accuracy)


##############################################################

##############################################################
'''
You Will be Required to record time for Training and Predicting 
The Code Given on Udacity Website is in Python-2
The Following Code is Python-3 version of the same code
'''

# t0 = time()
# # < your clf.fit() line of code >
# print("Training Time:", round(time()-t0, 3), "s")

# t0 = time()
# # < your clf.predict() line of code >
# print("Predicting Time:", round(time()-t0, 3), "s")

##############################################################