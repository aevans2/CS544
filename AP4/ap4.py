print(__doc__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from numpy import genfromtxt
from sklearn.metrics import accuracy_score

datainp = genfromtxt('https://drive.google.com/uc?export=download&id=0B490qbFJm8pxcEdDRkdWb3VkdjA', delimiter=',')

dataX = datainp[:,0:7]
dataY = datainp[:, 7].astype(int)

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]

X = dataX
y = dataY

estimator = {'k_means_iris_3': KMeans(n_clusters=3)}


fignum = 1
for name, est in estimator.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 4], X[:, 6], X[:, 3], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Kernel width')
    ax.set_ylabel('Groove length')
    ax.set_zlabel('Kernel length')
    fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('Kama', 1),
                    ('Rosa', 2),
                    ('Canadian', 3)]:
    ax.text3D(X[y == label, 4].mean(),
              X[y == label, 6].mean() + 1.5,
              X[y == label, 3].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y2 = np.choose(y, [0, 3, 1, 2])

ax.scatter(X[:, 4], X[:, 6], X[:, 3], c=y2)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Kernel width')
ax.set_ylabel('Groove length')
ax.set_zlabel('Kernel length')

#calculate accuracy
labels = np.choose(labels, [2, 3, 1, 0])
y_pred = labels
y_true = y

print ("Accuracy: ")
print(accuracy_score(y_true,y_pred))

print ("Coordinates of the cluster centroids: ")
print(est.cluster_centers_)

plt.show()
