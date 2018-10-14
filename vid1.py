# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 10:21:16 2018

@author: Daniel
"""

from sklearn import tree

clf = tree.DecisionTreeClassifier()

# CHALLENGE - create 3 more classifiers...
# KNN
# Random Forest
# SGD

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']


# CHALLENGE - ...and train them on our data
clf = clf.fit(X, Y)

newSamp = [[190, 70, 43]]
prediction = clf.predict(newSamp)

print(prediction)

''' new classifiers '''
#KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X,Y)
pred = knn.predict(newSamp)

print(pred)

#Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X,Y)
predrfc = knn.predict(newSamp)

print(predrfc)

#Linear Regression
from sklearn import linear_model

SGD = linear_model.SGDClassifier()
SGD.fit(X,Y)
predSGD = knn.predict(newSamp)

print(predSGD)
