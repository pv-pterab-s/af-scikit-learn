# -*- compile-command: ". ~/0.dev/bin/activate && . ~/workon_af.sh && cd ~/t-af-scikit-learn && python 1.af_test.py"; -*-
import numpy as np
from afsklearn.decomposition import PCA
import arrayfire as af
from arrayfire import display as D
from afsklearn.utils.validation import is_arrayfire_array
import time
import cProfile
import re
def Z(s):
    if is_arrayfire_array(s):
        print(s.to_ndarray())
    else:
        print(s)
M = print

# # test case 1
# X = af.from_ndarray(np.array([[-1.1, -1.2], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]]))
# pca = PCA(n_components=2)
# pca.fit(X)
# print("should get [0.99244289 0.00755711]...")
# print(pca.explained_variance_ratio_)
# print("should get [6.30061... 0.54980...]...")
# print(pca.singular_values_)

# A = np.load('501-input.npy')
n = 2**13
A = np.random.rand(n, n)

# A = np.array([[1,2],[3,4]])
af.set_backend('cpu')
af.info()
# A = af.from_ndarray(A).as_type(af.Dtype.f32)
pca = PCA(n_components=2)
tic = time.time()
# pca.fit(A)
cProfile.run('pca.fit(A)')
toc = time.time()
print(toc - tic)


# print("[1. 0.]")   # both cpu and gpu seem to work in tiny case
# Z(pca.explained_variance_ratio_)
# print("[2. 0.]")
# Z(pca.singular_values_)
