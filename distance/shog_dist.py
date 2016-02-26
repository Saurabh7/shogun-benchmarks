from time import time
import numpy as np
from modshogun import *

csv = np.genfromtxt('census_50k.csv', delimiter=",")

train_features = RealFeatures(csv.T)
distance = EuclideanDistance(train_features, train_features)

k = 10
mu = np.ones((train_features.get_num_features(), k))
mus = RealFeatures()
mus.copy_feature_matrix(mu)

t0 =time()
for i in range(100):
  distance.replace_rhs(mus)
  for j in range(train_features.get_num_vectors()):
    for m in range(k):
        distance.distance(j, m)
  mu = mu+1
  mus.copy_feature_matrix(mu)
  
t = time()-t0
print t
