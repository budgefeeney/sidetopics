# -*- coding: utf-8 -*-
'''
Implements the model from "Modelling Document Citations with
Latent Random Offsets"

@author: bryanfeeney
'''

__author__ = 'bryanfeeney'


from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as ssp
import numpy.random as rd

from util.array_utils import normalizerows_ip
from util.sigmoid_utils import rowwise_softmax, scaledSelfSoftDot, \
    colwise_softmax
from util.sparse_elementwise import sparseScalarQuotientOfDot, \
    sparseScalarQuotientOfNormedDot, sparseScalarProductOfSafeLnDot, \
    sparseScalarProductOfDot
from util.misc import printStderr, static_var
from util.overflow_safe import safe_log_det, safe_log
from model.evals import perplexity_from_like

from math import isnan

# ==============================================================
# CONSTANTS
# ==============================================================

DTYPE=np.float32 # A default, generally we should specify this in the model setup

DEBUG=False

MODEL_NAME="lro/vb"


# ==============================================================
# TUPLES
# ==============================================================

TrainPlan = namedtuple ( \
    'TrainPlan',
    'iterations epsilon logFrequency fastButInaccurate debug')

QueryState = namedtuple ( \
    'QueryState', \
    'outMeans outVarcs inMeans inVarcs inDocCov docLens'\
)

ModelState = namedtuple ( \
    'ModelState', \
    'K topicMean topicCov outDocCov vocab A trained dtype name'
)

# ==============================================================
# PUBLIC API
# ==============================================================
