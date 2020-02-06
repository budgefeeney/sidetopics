"""
This provides a Scikit-Learn compatible wrapper around all the models created
this far, to facilitate running them within the newer scikit-learn compatible
frameworks.
"""

from typing import Union
from types import ModuleType
import enum

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import numpy.random as rd
import math
import logging
from typing import List, Tuple, NamedTuple

from sidetopics.model.common import DataSet
import sidetopics.model.lda_cvb0 as _lda_cvb0
import sidetopics.model.lda_cvb as _lda_cvb
import sidetopics.model.lda_vb as _lda_vb
import sidetopics.model.lda_gibbs as _lda_gibbs
import sidetopics.model.lda_svb as _lda_svb
import sidetopics.model.lda_vb_python as _lda_vb_python


class TopicModelType(enum.Enum):
    LDA_CVB0 = _lda_cvb0
    LDA_CVB = _lda_cvb
    LDA_VB = _lda_vb
    LDA_VB_PYTHON_IMPL = _lda_vb_python
    LDA_SVB = _lda_svb  # the python one above has all the online algorithms
    LDA_GIBBS = _lda_gibbs



class EpochMetrics(NamedTuple):
    iters: List[int]
    bound_values: List[float]
    likelihood_values: List[float]

    @staticmethod
    def from_ibl_tuple(vals : Tuple[List[int], List[float], List[float]]) -> "EpochMetrics":
        return EpochMetrics(
            iters=None if len(vals) == 0 else vals[0],
            bound_values=None if len(vals) == 1 else vals[1],
            likelihood_values=None if len(vals) == 2 else vals[2],
        )

class TrainingStyleTypes(enum.Enum):
    BATCH = _lda_vb_python.RateAlgorBatch
    AMARIA = _lda_vb_python.RateAlgorAmaria
    BLEI_TIME_KAPPA = _lda_vb_python.RateAlgorTimeKappa
    VARIANCE = _lda_vb_python.RateAlgorVariance


class TopicModel(BaseEstimator, TransformerMixin):

    _module: ModuleType

    _model_state: _lda_cvb0.ModelState
    _last_query_state: _lda_cvb0.QueryState
    _last_X: DataSet
    _bound: float
    fit_metrics_: EpochMetrics
    transform_metrics_: EpochMetrics

    iterations: int
    query_iterations: int
    perp_tolerance: float
    use_approximations: bool

    n_components: int
    doc_topic_prior: Union[float, np.ndarray]
    topic_word_prior: Union[float, np.ndarray]

    # For online learning
    batchSize: int = 0  # 0 implies full-batch "batch" learning, rather than minibatch online learning
    rate_delay: float = 0.6
    forgetting_rate: float = 0.6
    rate_a: float = 2
    rate_b: float = 0.5
    rate_algor: TrainingStyleTypes = TrainingStyleTypes.BATCH

    # For sampling-based learning
    burn_in: int = -1
    thin: int = -1

    log_frequency: int
    debug: bool = False

    def __init__(self,
                 kind: TopicModelType,
                 n_components: int=10,
                 doc_topic_prior: Union[float, np.ndarray] = None,
                 topic_word_prior: Union[float, np.ndarray] = None,
                 iterations: int = 1000,
                 query_iterations: int = None,
                 perp_tolerance: float = None,
                 use_approximations: bool = False,
                 log_frequency: int = 0,
                 batchSize: int = 0,  # 0 implies full-batch "batch" learning, rather than minibatch online learning
                 rate_delay: float = 0.6,
                 forgetting_rate: float = 0.6,
                 rate_a: float = 2,
                 rate_b: float = 0.5,
                 rate_algor: TrainingStyleTypes = TrainingStyleTypes.BATCH,
                 burn_in: int = -1,
                 thin: int = -1,
                 debug: bool = False,
                 seed: int = 0xC0FFEE):
        self._module = kind.value

        self._model_state = None
        self._last_query_state = None
        self._last_X = None
        self._bound = None

        self.n_components = n_components
        self.doc_topic_prior = doc_topic_prior
        self.topic_word_prior = topic_word_prior
        self.bound_ = math.nan

        self.iterations = iterations
        self.query_iterations = query_iterations or self.iterations // 10
        self.perp_tolerance = perp_tolerance or 0.1
        self.use_approximations = use_approximations
        self.log_frequency = log_frequency

        self.batchSize = batchSize
        self.rate_delay = rate_delay
        self.forgetting_rate = forgetting_rate
        self.rate_a = rate_a
        self.rate_b = rate_b
        self.rate_algor = rate_algor

        self.burn_in = burn_in
        self.thin = thin

        self.debug = debug
        rd.seed(seed)
        if self._module is _lda_gibbs:
            self._module.seed_rng(seed)

    def copy(self) -> "TopicModel":
        """
        Slightly shallow copy, last query state is not amended
        :return:
        """
        result = TopicModel()
        result.set_params(**result.get_params())
        result._model_state = _lda_cvb0.newModelFromExisting(self._model_state)
        result._last_query_state = self._last_query_state

    def fit_transform(self,  X: DataSet, y: np.ndarray = None, **kwargs) -> np.ndarray:
        return self.fit(X, y, **kwargs).transform(X, **kwargs)

    def fit(self,
            X: DataSet,
            y: np.ndarray = None,
            **kwargs) -> "LdaCvb":
        """
        Fits the current model. If called twice, will use the previous _trained_ model
        state as the current initial value to refine, i.e. it';; train further
        :param X: the document statistics, and associated features
        :param y: Not used, part of the sklearn API
        :param iterations: how many iterations to train for
        :param kwargs: Any futher (undocumented) arguments.
        :return: a reference to this model object
        """
        module = self._module
        if self._model_state is None:
            logging.info("Creating new model-state at random")
            self._model_state = module.newModelAtRandom(data=X,
                                                        topicPrior=self.doc_topic_prior,
                                                        vocabPrior=self.topic_word_prior,
                                                        K=self.n_components)
        iters = kwargs.get('iterations') or self.iterations
        train_plan = self._new_train_plan(module, iters, **kwargs)
        input_query = self.make_or_resume_query_state(X, kwargs.get('resume'))

        self._model_state, self._last_query_state, m = module.train(X, self._model_state, input_query, train_plan)
        self.bound_ = math.nan
        self.fit_metrics_ = EpochMetrics.from_ibl_tuple(m)
        return self

    def _new_train_plan(self, module, iters: int, **kwargs):
        if module is _lda_vb_python:
            return module.newTrainPlan(
                iterations=iters,
                epsilon=self.perp_tolerance,
                logFrequency=self.log_frequency,
                fastButInaccurate=self.use_approximations,
                debug=self.debug,
                batchSize=self.batchSize,
                rate_delay=self.rate_delay,
                forgetting_rate=self.forgetting_rate,
                rate_a=self.rate_a,
                rate_b=self.rate_b,
                rate_algor=self.rate_algor.value
            )
        elif module is _lda_gibbs:
            return module.newTrainPlan(
                iterations=iters,
                burnIn=kwargs.get('burn_in') or self.burn_in,
                thin=kwargs.get('thin') or self.thin,
                logFrequency=self.log_frequency,
                fastButInaccurate=self.use_approximations,
                debug=self.debug
            )
        else:
            return module.newTrainPlan(
                iterations=iters,
                epsilon=self.perp_tolerance,
                logFrequency=self.log_frequency,
                fastButInaccurate=self.use_approximations,
                debug=self.debug
            )

    def transform(self,
            X: DataSet,
            y: np.ndarray = None,
            **kwargs) -> np.ndarray:
        """
        Transforms the given data: essentially inferring topic-assignment distributions for the data keeping
        all other model parameters (word distributions, topic-priors) fixed.

        If "resume" is true, then we assume this is the _second_ time training on the given dataset, and
        use the past set of feature-assignments as the initial state to refine given the data. This is
        a strange thing to do
        :param X: the document statistics, and associated features
        :param y: Not used, part of the sklearn API
        :param iterations: how many iterations to run in the query phase
        :param kwargs: Any futher (undocumented) arguments.
        :return: a distribution of topic-assignment probabilities one for each document.
        """
        module = self._module
        if self._model_state is None:
            raise ValueError("Untrained model")

        input_query = self.make_or_resume_query_state(X, kwargs.get('resume'))

        iters = kwargs.get('iterations') or self.query_iterations
        query_plan = self._new_train_plan(module, iters, **kwargs)

        self._model_state, inferred_query_state = module.query(X, self._model_state, input_query, query_plan)

        self.transform_metrics_ = None
        if kwargs.get('persist_query_state', True):
            self._bound = math.nan
            self._last_query_state = inferred_query_state

        return self._module.topicDists(self._last_query_state)

    def make_or_resume_query_state(self, X, should_resume):
        if should_resume:
            logging.info("Resuming from last query state")
            input_query = self._last_query_state
        else:
            logging.info("Creating new query state")
            input_query = self._module.newQueryState(X, self._model_state, self.debug)
        return input_query

    @property
    def components_(self):
        return self._module.wordDists(self._last_query_state)

    def score(self, X: DataSet, y: np.ndarray = None, persist_query_state: bool = False, method="varbound", **kwargs) -> float:
        """
        Infers topic weights for the given data, holding the components
        (i.e. vocabularies) fixed.

        Returns the variational lower-bound calculated on the given data,
        which is an approximation of the true log-likelihood.
        """
        if method == "loglikelihood":
            return self.score_with_point_log_likelihood(X, y, persist_query_state, **kwargs)
        elif method == "perplexity":
            return self.score_with_perplexity(X, y, persist_query_state, **kwargs)
        elif method == "doc-completion-perplexity":
            return self.score_with_doc_completion_perplexity(X, y, persist_query_state, **kwargs)
        elif method is not None and method != 'varbound':
            raise ValueError(f"Unknown scoring method {method}")

        old_last_query_state = self._last_query_state
        if X is not self._last_X:  # an an optimisation, don't query twice if it's the exact same *object*
            _ = self.transform(X, y, persist_query_state=True)

        bound = self._module.var_bound(X, self._model_state, self._last_query_state)
        if not persist_query_state:
            self._last_query_state = old_last_query_state

        return bound

    def score_with_point_log_likelihood(self, X: DataSet, y: np.ndarray = None, persist_query_state: bool = False, **kwargs) -> float:
        """
        Infers topic weights for the given data, holding the components
        (i.e. vocabularies) fixed.

        Returns the approximate, point-estimated, log-likelihood
        """
        old_last_query_state = self._last_query_state
        if X is not self._last_X:  # an an optimisation, don't query twice if it's the exact same *object*
            _ = self.transform(X, y, persist_query_state=True)

        bound = self._module.log_likelihood(X, self._model_state, self._last_query_state)
        if not persist_query_state:
            self._last_query_state = old_last_query_state

        return bound

    def score_with_perplexity(self, X: DataSet, y: np.ndarray = None, persist_query_state: bool = False, **kwargs) -> float:
        """
        Infers topic weights for the given data, holding the components
        (i.e. vocabularies) fixed.

        Calcuates the approximate, point-estimated, log-likelihood; then
        uses that to calculate the perplexit.
        """

        return self._module.perplexity_from_like(
            self.score_with_point_log_likelihood(X, y, persist_query_state),
            np.sum(X.words, axis=1)
        )

    def score_with_doc_completion_perplexity(self, X: DataSet, y: np.ndarray = None, persist_query_state: bool = False, **kwargs) -> float:
        """
        Infers topic weights for the half the symbols in the given data, holding the
        components (i.e. vocabularies) fixed.

        Calcuates the approximate, point-estimated, log-likelihood of the other
        half with the components and loadings; then uses that to calculate the
        perplexity.
        """

        return self._module.perplexity_from_like(
            self.score_with_point_log_likelihood(X, y, persist_query_state),
            np.sum(X.words, axis=1)
        )
