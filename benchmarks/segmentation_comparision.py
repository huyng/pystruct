import numpy as np
import cPickle
#import matplotlib.pyplot as plt

from pystruct.models import EdgeFeatureGraphCRF
from pystruct.learners import FrankWolfeSSVM, SubgradientSSVM, NSlackSSVM, OneSlackSSVM
from pystruct.utils import AnalysisLogger


from sklearn.utils import shuffle

data_train = cPickle.load(open("msrc_dict_train.pickle"))
C = 0.01

n_states = 21
n_samples = len(data_train['X'])
print("number of samples: %s" % n_samples)
class_weights = 1. / np.bincount(np.hstack(data_train['Y']))
class_weights *= 21. / np.sum(class_weights)
# Train linear chain CRF
#model = ChainCRF(inference_method=('ogm', {'alg': 'dyn'}))
model = EdgeFeatureGraphCRF(inference_method='qpbo',
                                 class_weight=class_weights,
                                 symmetric_edge_features=[0, 1],
                                 antisymmetric_edge_features=[2])

bcfw = FrankWolfeSSVM(model=model, C=.01, max_iter=100000, tol=0.1, verbose=3, check_dual_every=10, averaging='linear')
pegasos = SubgradientSSVM(model=model, C=.01, max_iter=1000, verbose=3, momentum=0, decay_exponent=1, decay_t0=1, learning_rate=n_samples * .01)
nslack = NSlackSSVM(model, C=.01, tol=.01, verbose=3)
nslack_every = NSlackSSVM(model, C=.01, tol=.1, verbose=3, batch_size=1)
oneslack = OneSlackSSVM(model, C=.01, tol=.1, verbose=3)
oneslack_cache = OneSlackSSVM(model, C=.01, tol=.1, inference_cache=50, verbose=3)

#svms = [bcfw, oneslack, oneslack_cache, pegasos, nslack, nslack_every]
#names = ['bcfw', "oneslack", "oneslack_cache", "pegasos", "nslack", "nslack_every"]
svms = [bcfw, oneslack, oneslack_cache, pegasos]
names = ['bcfw', "oneslack", "oneslack_cache", "pegasos"]

X_train, Y_train = shuffle(data_train['X'], data_train['Y'])

for name, svm in zip(names, svms):
    logger = AnalysisLogger("msrc_" + name + ".pickle", log_every=10)
    svm.logger = logger
    svm.fit(X_train, Y_train)
