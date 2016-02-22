from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
from time import time
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values="NaN", strategy='mean', axis=0)

csv = np.genfromtxt('census_50k.csv', delimiter=",")
#imp.fit(csv)
#imp.transform(csv)
#print np.any(np.isnan(csv))
t0 = time()
k = KMeans(init='random', n_clusters=1000, n_init=1)
k.fit(csv)
t = time()-t0
print t
