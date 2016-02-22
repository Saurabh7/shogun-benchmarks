from time import time
import numpy as np
from modshogun import *

k = 100

csv = np.genfromtxt('student.csv', delimiter=",")

t0 = time()

train_features = RealFeatures(csv.T)
distance = EuclideanDistance(train_features, train_features)

kmeans = KMeans(k, distance)
kmeans.train()

t = time()-t0
print t

