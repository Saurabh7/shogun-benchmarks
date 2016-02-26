from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
from time import time
from sklearn.utils.extmath import row_norms, squared_norm
from sklearn.metrics.pairwise import euclidean_distances


k=10
csv = np.genfromtxt('census_50k.csv', delimiter=",")
sh = csv.shape
mu = np.ones((k, sh[1]))
x_squared_norms = row_norms(csv, squared=True)

t0 = time()
for i in range(100):
  all_distances = euclidean_distances(mu, csv, x_squared_norms,
                                        squared=True)
  mu = mu + 1

t = time() - t0
print t
