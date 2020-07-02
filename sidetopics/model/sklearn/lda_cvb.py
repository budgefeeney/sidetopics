"""
This provides a Scikit-Learn compatible wrapper around all the models created
this far, to facilitate running them within the newer scikit-learn compatible
frameworks.
"""

from typing import Union, Dict
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
import sidetopics.model.mom_em as _mom_em
import sidetopics.model.mom_gibbs as _mom_gibbs


_ = logging.getLogger()
logging.basicConfig(level=logging.DEBUG,
                    format="%(filename)s:%(funcName)s:%(lineno)s   :: %(message)s")


def dict_without(d: Dict[str, str], key: str) -> Dict[str, str]:
    """
    A copy of the dict with everything except the given key
    """
    r = d.copy()
    if key in r:
        r.pop(key)
    return r


class ScoreMethod(enum.Enum):
    LogLikelihoodPoint = 'log_likelihood_point'
    LogLikelihoodBoundOrSampled = 'log_likelihood_expectation'
    PerplexityPoint = 'perplexity_point'
    PerplexityBoundOrSampled = 'perplexity_expectation'

    @staticmethod
    def from_str(name: str) -> "ScoreMethod":
        name = name.strip().lower()
        for possibility in ScoreMethod:
            if name == possibility.value:
                return possibility
        raise ValueError("No score methodd with that name")
    
    def is_point_estimate(self) -> bool:
        return self in [ScoreMethod.LogLikelihoodPoint, ScoreMethod.PerplexityPoint]
    
    def is_perplexity(self) -> bool:
        return self in [ScoreMethod.PerplexityBoundOrSampled, ScoreMethod.PerplexityPoint]


class TopicModelType(enum.Enum):
    LDA_CVB0 = _lda_cvb0
    LDA_CVB = _lda_cvb
    LDA_VB = _lda_vb
    LDA_VB_PYTHON_IMPL = _lda_vb_python
    LDA_SVB = _lda_svb  # the python one above has all the online algorithms
    LDA_GIBBS = _lda_gibbs
    MOM_VB = _mom_em
    MOM_GIBBS = _mom_gibbs

    def uses_sampling_based_inference(self) -> bool:
        return (self is TopicModelType.MOM_GIBBS) or (self is TopicModelType.LDA_GIBBS)




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
    kind: TopicModelType

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
        self.kind = kind

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
        return self.fit(
            X, y,
            persist_query_state=True,
            **dict_without(kwargs, 'persist_query_state')
        ).transform(
            X,
            resume=True,
            **dict_without(kwargs, 'resume')
        )

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
        :param persist_query_state: set to true to be able to continue where you left off on a second
        call to transform
        :param resume: if True use the previous output from either train() or transform(). Only works
        if you set persist_query_state=True in the previous call to either train() or transform()
        as the initial starting state.
        :param kwargs: Any further (undocumented) arguments.
        :return: a reference to this model object
        """
        module = self._module
        if self._model_state is None:
            logging.info("Creating new model-state at random")
            self._model_state = module.newModelAtRandom(data=X,
                                                        topicPrior=self.doc_topic_prior,
                                                        vocabPrior=self.topic_word_prior,
                                                        K=self.n_components)
        else:
            logging.info("Resuming fit using previous model state")

        iters = kwargs.get('iterations') or self.iterations
        train_plan = self._new_train_plan(module, iters, **kwargs)
        input_query = self.make_or_resume_query_state(X, kwargs.get('resume'))

        self._model_state, inferred_query_state, m = module.train(X, self._model_state, input_query, train_plan)
        self.bound_ = math.nan
        self._set_or_clear_last_query_state(X, inferred_query_state, kwargs.get('persist_query_state'))

        self.fit_metrics_ = EpochMetrics.from_ibl_tuple(m)
        logging.info(f"Fit model to data after {iters} iterations")
        return self

    def _set_or_clear_last_query_state(self, X: DataSet, inferred_query_state, persist_query_state=False):
        if persist_query_state:
            logging.info("Persisting query state")
            self._last_X = X
            self._last_query_state = inferred_query_state
        else:
            logging.info("Resetting persisted query state to None")
            self._last_X = None
            self._last_query_state = None

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
        use the past set of feature-assignments as the initial state to refine given the data. This is to
        transform more.... An ideal way of doing this woule be just to pass in an initial transform as an
        input, but expanding that into variational distributions, or past samples, would be too much. So
        we just do this hack where the class retains the state of the last transformed data in order to
        be able to continue from where we left off.

        :param X: the document statistics, and associated features
        :param y: Not used, part of the sklearn API
        :param iterations: how many iterations to run in the query phase
        :param persist_query_state: set to true to be able to continue where you left off on a second
        call to transform
        :param resume: set to true to continue where you left off on a second call to transform. Requires
        that you set persist_query_state to True in the previous round.
        :param kwargs: Any futher (undocumented) arguments.
        :return: a distribution of topic-assignment probabilities one for each document.
        """
        logging.debug("Transforming data")
        module = self._module
        if self._model_state is None:
            raise ValueError("Untrained model")

        input_query = self.make_or_resume_query_state(X, kwargs.get('resume'))

        iters = kwargs.get('iterations') or self.query_iterations
        query_plan = self._new_train_plan(module, iters, **kwargs)

        self._model_state, inferred_query_state = module.query(X, self._model_state, input_query, query_plan)

        self.transform_metrics_ = None
        self._bound = math.nan
        self._set_or_clear_last_query_state(X, inferred_query_state, kwargs.get('persist_query_state'))

        ret_val = self._module.topicDists(inferred_query_state)
        logging.info(f"Obtained a {ret_val.shape} topic-assignments matrix after {iters} iterations")
        return ret_val

    def make_or_resume_query_state(self, X: DataSet, should_resume: bool):
        if should_resume:
            if self._last_X is X:
                logging.info("Resuming from last query state")
                input_query = self._last_query_state
            else:
                raise ValueError("Attempting to resume with a different dataset to last fit/transform run")
        else:
            logging.info("Creating new query state")
            input_query = self._module.newQueryState(X, self._model_state, self.debug)
        return input_query

    @property
    def components_(self):
        return self._module.wordDists(self._model_state)

    @property
    def query_state_(self):
        return self._last_query_state

    def score(self, X: DataSet, y: np.ndarray = None, y_query_state: object = None, method: Union[ScoreMethod, str] = ScoreMethod.LogLikelihoodPoint, **kwargs) -> float:
        """
        Uses the given topic assignments `y` to determine the fit of the given model according
        to either log-likelihood, or perplexity (which is a function of log likelihood).

        The log likelihood can be determined either via a point estimate (substituting in the
        _mean_ of the parameter posteriors into the likelihood equataion) or by taking the
        full joint expectation of the data and all parameters, marginalising out the uncertainty.

        In the case where no assignments `y` are provided, the prior (which may have been
        learnt from the data) is used instead.

        Instead of `y` you can instead provide the QueryState object (see the query_state_
        property) as `y_query_state`. It is mandatory to provide this for topic-model
        types that use sampling based inference. It is an error to provide both `y` and
        `y_query_state`. It is a okay to provide neither: in that case the prior over
        topics will be used, i.e. the stanard equation for the log-likelihood in a
        mixture model.

        :param X: the data to score with this model
        :param y: the topic assignments obtained with `fit` which we're using to score the
        data. Alternatively you can provide y_query_state instead, which is required for
        cases where `TopicModel.kind.uses_sampling_based_inference() == True`
        :param y_query_state: the query-state obtained by calling the `fit` function with
        `perist_query_state=True` and then accessing the `query_state_` property. This
        provide additional information about the uncertainty of y and is vital for when
        `TopicModel.kind.uses_sampling_based_inference() == True`
        :param method the scoring method to use. If a string, we delegate to `ScoreMethod.from_str()`
        """
        if type(method) is str:
            method = ScoreMethod.from_str(method)

        if (y is not None) and (y_query_state is not None):
            raise ValueError("Cannot specify both y and y_query_state at the same time")

        if (y is None) and (y_query_state is None):
            logging.warning("Using prior topic assignments to score data according to model.")
            query_state = None
        elif y is None:
            query_state = y_query_state
        elif y_query_state is None:
            if self.kind.uses_sampling_based_inference():
                if not method.is_point_estimate():
                    raise ValueError("For a Gibbs-sampling based model, you can only use an externally sourced topic "
                                     "assignments with point-estimated scores. Transform/fit with persist_query_state ="
                                     " True  to make query-state available for call to score")
            logging.info("Creating new query state with given assignments (skipping transform step) for scoring")
            query_state = self.make_or_resume_query_state(X, should_resume=False)._replace(topicDists=y)


        if method.is_point_estimate():
            logging.info("Obtaining point estimate of log-likelihodd")
            log_prob = self._module.log_likelihood_point(X, self._model_state, query_state).sum()
        else:
            logging.info("Obtaining expected value (samples, variational bound) of log likelihood")
            log_prob = self._module.log_likelihood_expected(X, self._model_state, query_state).sum()
            
        if method.is_perplexity():
            logging.info("Converting log likelihood to perplexity")
            return self._module.perplexity_from_like(
                log_prob,
                X.word_count
            )
        else:
            return log_prob

    def find_query_state_via_transform(self,
                                       X: DataSet,
                                       transform_kwargs: Dict[str, str],
                                       persist_query_state: bool) -> object:
        last_query_state_backup, last_X_backup = None, None
        if not persist_query_state and (self._last_query_state is not None):
            last_query_state_backup = self._last_query_state
            last_X_backup = self._last_X

        _ = self.transform(X, persist_query_state=True, **dict_without(transform_kwargs, 'persist_query_state'))
        query_state = self._last_query_state

        if not persist_query_state:
            self._last_query_state = last_query_state_backup
            self._last_X = last_X_backup

        return query_state
            
        
