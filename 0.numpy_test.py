# -*- compile-command: "source ~/0.dev/bin/activate && source ~/workon_af.sh && cd ~/af-scikit-learn && python 0.numpy_test.py"; -*-
import numpy as np
from afsklearn.decomposition import PCA
import arrayfire as af
M = print

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)

print(pca.explained_variance_ratio_)
# [0.9924... 0.0075...]
print(pca.singular_values_)
# [6.30061... 0.54980...]
