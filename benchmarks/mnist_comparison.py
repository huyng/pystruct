import numpy as np
from sklearn.datasets import fetch_mldata

from pystruct.models import MultiClassClf
from pystruct.learners import FrankWolfeSSVM, SubgradientSSVM, NSlackSSVM, OneSlackSSVM
from pystruct.utils import AnalysisLogger

from sklearn.utils import shuffle

mnist = fetch_mldata("MNIST original")
X_train, y_train = mnist.data[:60000], mnist.target[:60000]

X_train = X_train / 255.
y_train = y_train.astype(np.int)

model = MultiClassClf()

bcfw = FrankWolfeSSVM(model=model, C=.1, max_iter=10000, tol=0.1, verbose=3, check_dual_every=10, averaging='linear')
pegasos = SubgradientSSVM(model=model, C=.1, max_iter=1000, verbose=3, momentum=0, decay_exponent=1, decay_t0=1, learning_rate=len(X_train) * .1)
nslack = NSlackSSVM(model, C=.1, tol=.1, verbose=3)
nslack_every = NSlackSSVM(model, C=.1, tol=.1, verbose=3, batch_size=1)
oneslack = OneSlackSSVM(model, C=.1, tol=.1, verbose=3)
oneslack_cache = OneSlackSSVM(model, C=.1, tol=.1, inference_cache=50, verbose=3)

#svms = [bcfw, oneslack, oneslack_cache, pegasos, nslack, nslack_every]
#names = ['bcfw', "oneslack", "oneslack_cache", "pegasos", "nslack", "nslack_every"]

svms = [bcfw]
names = ['bcfw3']

X_train, y_train = shuffle(X_train, y_train)

for name, svm in zip(names, svms):
    logger = AnalysisLogger("mnist_" + name + ".pickle", log_every=10)
    svm.logger = logger
    svm.fit(X_train, y_train)
