# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:18:19 2017

@author: LiuYX
"""

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

iris = datasets.load_iris()
digits = datasets.load_digits()
iris_X = iris.data
iris_y = iris.target

#print(iris_X[::50])
#print(iris_y[::50])

#train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size=0.2) 
#
#knn = KNeighborsClassifier()
#knn.fit(train_X, train_y)
#
#predicts = knn.predict(test_X)
#print(predicts)
#print(test_y)
#print(knn.score(test_X, test_y))
X = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [0, 0]]
y = [0, 1, 2, 3, 4, 0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)

from sklearn.externals.six import StringIO
import pydot 
dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("iris.pdf") 
#with open("iris.dot", 'w') as f:
#    f = tree.export_graphviz(clf, out_file=f)
#import os
#os.unlink('iris.dot')