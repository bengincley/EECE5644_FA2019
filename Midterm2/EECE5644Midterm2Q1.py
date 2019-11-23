# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:12:55 2019

@author: bengi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pydot
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from mlxtend.plotting import plot_decision_regions


def confusionMatrixFxn(labelsTest,prediction):
    ##  Plot normalized confusion matrix
    confusionMatrix = metrics.confusion_matrix(labelsTest,prediction)
    normalized_CM = np.zeros((2,2))
    true_totals = np.sum(confusionMatrix,1)
    for i in range(len(true_totals)):
        normalized_CM[:,i] = confusionMatrix[:,i]/true_totals
    classes=unique_labels
    cm=plt.cm.Blues
    fig, ax = plt.subplots()
    im = ax.imshow(normalized_CM, interpolation='nearest', cmap=cm)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusionMatrix.shape[1]),
           yticks=np.arange(confusionMatrix.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Normalized confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')
    fmt = '.2f'
    for i in range(normalized_CM.shape[0]):
        for j in range(normalized_CM.shape[1]):
            ax.text(j, i, format(normalized_CM[i, j], fmt),
                    ha="center", va="center",
                    color="white" if normalized_CM[i, j] > 0.5 else "black")
    fig.tight_layout()

##  Load data
columns = list(['x1','x2','labels'])
data = pd.read_csv('C:/Users/bengi/OneDrive - Northeastern University/1. Graduate/9. Courses/EECE5644 Machine Learning/Midterm Exam/Exam2/Q1.csv',names=columns)
labels = np.array(data['labels'])
unique_labels = np.unique(labels)
labelNames = list(['-1','1'])
x1 = data['x1']
x2 = data['x2']
x = data[['x1','x2']]

##  Train Decision Tree model
xTrain, xTest, labelsTrain, labelsTest = train_test_split(x,labels,test_size=0.1,random_state=1)
decTree = DecisionTreeClassifier(criterion="entropy",max_depth=12,min_impurity_decrease=0.01,max_leaf_nodes=12)
decTree.fit(xTrain,labelsTrain)

decTreePredict = decTree.predict(xTest)

errors = np.not_equal(decTreePredict,labelsTest)
nErrors = sum(errors)
print("Accuracy of decision tree is: "+str(100*metrics.accuracy_score(labelsTest,decTreePredict)))

export_graphviz(decTree, out_file='Q2tree.dot', feature_names = columns[0:2], rounded = True, precision = 1, class_names = labelNames, label = 'all', filled = True)
(graph, ) = pydot.graph_from_dot_file('Q2tree.dot')
graph.write_png('Q2tree.png')

report1 = metrics.classification_report(labelsTest,decTreePredict)
#print(report1)

confusionMatrixFxn(labelsTest,decTreePredict)

##  Plot data figure
plt.figure()
scatter1, = plt.plot(x1[labels==-1],x2[labels==-1],'o',color=((1,0,0)),markersize=5)
scatter1.set_label('Class -1')
scatter2, = plt.plot(x1[labels==1],x2[labels==1],'x',color=((0,0,0)),markersize=8)
scatter2.set_label('Class 1')
plt.title('Measurements x1 and x2 for Classes -1 and 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='lower right')
plt.grid()

##  Plot the decision boundaries
y = np.asarray(labelsTest).reshape(len(labelsTest))
x = np.asarray(xTest)
plot_decision_regions(X=x, y=y, clf=decTree)
plt.title('Decision boundary region for decision tree')
plt.show()

##
##  Create bagging model
##
baggedTrees = BaggingClassifier(base_estimator=decTree, n_estimators = 7, max_samples=1.0, bootstrap=True, random_state = 1)
baggedTrees.fit(xTrain,labelsTrain)

bagPredict = baggedTrees.predict(xTest)
report2 = metrics.classification_report(labelsTest,bagPredict)
#print(report2)
print("Accuracy of bagged model is: "+str(100*metrics.accuracy_score(labelsTest,bagPredict)))

confusionMatrixFxn(labelsTest,bagPredict)

##  Plot the decision boundaries
y = np.asarray(labelsTest).reshape(len(labelsTest))
x = np.asarray(xTest)
plot_decision_regions(X=x, y=y, clf=baggedTrees)
plt.title('Decision boundary region for Bagging')
plt.show()

##
## Adaboost Classifier
##
for i in range(1,20):

    boostedTrees = AdaBoostClassifier(base_estimator=decTree, n_estimators = 7, learning_rate=i*0.1, algorithm='SAMME.R')
    boostedTrees.fit(xTrain,labelsTrain)
    
    boostPredict = boostedTrees.predict(xTest)
    #report3 = metrics.classification_report(labelsTest,boostPredict)
#    print(i*0.1)
#    print("Accuracy of boosted model is: "+str(100*metrics.accuracy_score(labelsTest,boostPredict)))
    
optimalBoostedTrees = AdaBoostClassifier(base_estimator=decTree, n_estimators = 7, learning_rate=1.4, algorithm='SAMME.R')
optimalBoostedTrees.fit(xTrain,labelsTrain)

boostPredict = optimalBoostedTrees.predict(xTest)
#report3 = metrics.classification_report(labelsTest,boostPredict)
print("Accuracy of boosted model is: "+str(100*metrics.accuracy_score(labelsTest,boostPredict)))
weights = optimalBoostedTrees.estimator_weights_

confusionMatrixFxn(labelsTest,boostPredict)

##  Plot the decision boundaries
y = np.asarray(labelsTest).reshape(len(labelsTest))
x = np.asarray(xTest)
plot_decision_regions(X=x, y=y, clf=optimalBoostedTrees)
plt.title('Decision boundary region for Adaboost')
plt.show()