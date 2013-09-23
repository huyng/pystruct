import numpy as np
from scipy import sparse
from sklearn.datasets import fetch_mldata

from pystruct.models import MultiLabelClf
from pystruct.learners import FrankWolfeSSVM, SubgradientSSVM, NSlackSSVM, OneSlackSSVM
from pystruct.utils import AnalysisLogger
from pystruct.datasets import load_scene
from sklearn.metrics import mutual_info_score
from sklearn.utils import minimum_spanning_tree

import itertools
from sklearn.utils import shuffle

def chow_liu_tree(y_):
    # compute mutual information using sklearn
    n_labels = y_.shape[1]
    mi = np.zeros((n_labels, n_labels))
    for i in xrange(n_labels):
        for j in xrange(n_labels):
            mi[i, j] = mutual_info_score(y_[:, i], y_[:, j])
    mst = minimum_spanning_tree(sparse.csr_matrix(-mi))
    edges = np.vstack(mst.nonzero()).T
    edges.sort(axis=1)
    return edges

dataset = "scene"
#dataset = "yeast"

if dataset == "yeast":
    yeast = fetch_mldata("yeast")

    X = yeast.data
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    y = yeast.target.toarray().astype(np.int).T

    X_train, X_test = X[:1500], X[1500:]
    y_train, y_test = y[:1500], y[1500:]

else:
    scene = load_scene()
    X_train, X_test = scene['X_train'], scene['X_test']
    y_train, y_test = scene['y_train'], scene['y_test']

n_labels = y_train.shape[1]
full = np.vstack([x for x in itertools.combinations(range(n_labels), 2)])
tree = chow_liu_tree(y_train)

model = MultiLabelClf(edges=full, inference_method=('ad3', {'branch_and_bound': True}))
#independent_model = MultiLabelClf(inference_method='unary')
#model = MultiLabelClf(edges=tree, inference_method=('ogm', {'alg':'dyn'}))

bcfw = FrankWolfeSSVM(model=model, C=.1, max_iter=1000, tol=0.1, verbose=3, check_dual_every=10, averaging='linear')
pegasos = SubgradientSSVM(model=model, C=.1, max_iter=1000, verbose=3, momentum=0, decay_exponent=1, decay_t0=1, learning_rate=len(X_train) * .1)
nslack = NSlackSSVM(model, C=.1, tol=.1, verbose=3)
nslack_every = NSlackSSVM(model, C=.1, tol=.1, verbose=3, batch_size=1)
oneslack = OneSlackSSVM(model, C=.1, tol=.1, verbose=3)
oneslack_cache = OneSlackSSVM(model, C=.1, tol=.1, inference_cache=50, verbose=3)

#svms = [bcfw, oneslack, oneslack_cache, pegasos, nslack, nslack_every]
#names = ['bcfw', "oneslack", "oneslack_cache", "pegasos", "nslack", "nslack_every"]
svms = [bcfw, oneslack_cache, nslack, nslack_every]
names = ['SZLJSP', "1-cache-slack", "n-slack", "n-slack-every"]

X_train, y_train = shuffle(X_train, y_train)

for name, svm in zip(names, svms):
    logger = AnalysisLogger("scene_full_" + name + ".pickle", log_every=10)
    svm.logger = logger
    svm.fit(X_train, y_train)
