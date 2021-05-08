"""
This provides a Scikit-Learn compatible wrapper around all the models created
this far, to facilitate running them within the newer scikit-learn compatible
frameworks.
"""

from typing import Union, Dict, Optional
from types import ModuleType
import enum

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
import numpy as np
import numpy.random as rd
import scipy.sparse as ssp
import math
import logging
from typing import List, Tuple, NamedTuple
import importlib

from sidetopics.model.common import DataSet
import sidetopics.model.lda_cvb0 as _lda_cvb0
import sidetopics.model.lda_cvb as _lda_cvb
import sidetopics.model.lda_vb as _lda_vb
import sidetopics.model.lda_gibbs as _lda_gibbs
import sidetopics.model.lda_svb as _lda_svb
import sidetopics.model.lda_vb_python as _lda_vb_python
import sidetopics.model.mom_em as _mom_em
import sidetopics.model.mom_gibbs as _mom_gibbs
from gensim.sklearn_api import HdpTransformer
from gensim.corpora.dictionary import Dictionary
from gensim import matutils

from sidetopics.util.sparse_elementwise import sparseScalarProductOfSafeLnDot
from sidetopics.util.overflow_safe import safe_log
from sklearn.decomposition import LatentDirichletAllocation
from numpy.random import RandomState

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
    DocCompletionLogLikelihoodPoint = 'doc_completion_log_likelihood_point'
    DocCompletionPerplexityPoint = 'doc_completion_perplexity_point'

    @staticmethod
    def from_str(name: str) -> "ScoreMethod":
        name = name.strip().lower()
        for possibility in ScoreMethod:
            if name == possibility.value:
                return possibility
        raise ValueError("No score method with that name")

    def is_point_estimate(self) -> bool:
        return self in [ScoreMethod.LogLikelihoodPoint,
                        ScoreMethod.PerplexityPoint,
                        ScoreMethod.DocCompletionLogLikelihoodPoint,
                        ScoreMethod.DocCompletionPerplexityPoint
                       ]

    def is_perplexity(self) -> bool:
        return self in [ScoreMethod.PerplexityBoundOrSampled,
                        ScoreMethod.PerplexityPoint,
                        ScoreMethod.DocCompletionPerplexityPoint
                       ]

    def is_doc_completion(self) -> bool:
        return self in [ScoreMethod.DocCompletionLogLikelihoodPoint,
                        ScoreMethod.DocCompletionPerplexityPoint
                        ]


class TopicModelType(enum.Enum):
    LDA_CVB0 = _lda_cvb0
    LDA_CVB = _lda_cvb
    LDA_VB = _lda_vb
    LDA_VB_PYTHON_IMPL = _lda_vb_python
    LDA_SVB = _lda_svb  # the python one above has all the online algorithms
    LDA_GIBBS = _lda_gibbs
    MOM_VB = _mom_em
    MOM_GIBBS = _mom_gibbs

    def uses_bayesian_inference(self) -> bool:
        return True

    def is_mixture(self) -> bool:
        return (self is TopicModelType.MOM_GIBBS) or (self is TopicModelType.MOM_VB)

    def is_admixture(self) -> bool:
        return not self.is_mixture()


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
    kind_name: str

    _model_state: object
    _last_query_state: object
    _last_X: Optional[DataSet]
    _bound: Optional[float]
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
    batchSize: int
    rate_delay: float
    forgetting_rate: float
    rate_a: float
    rate_b: float
    rate_algor: TrainingStyleTypes

    # For sampling-based learning
    burn_in: int
    thin: int

    log_frequency: int
    debug: bool

    def __init__(self,
                 kind: Union[str, TopicModelType],
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
                 default_scoring_method: ScoreMethod = ScoreMethod.LogLikelihoodPoint,
                 debug: bool = False,
                 seed: int = 0xC0FFEE):
        if type(kind) is TopicModelType:
            self.kind = kind.name
        else:
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
        self.default_scoring_method = default_scoring_method

        self.seed = seed
            
    
    def __setattr__(self, name, value) -> None:
        if name == 'seed':
            rd.seed(value)
            if TopicModelType[self.kind] is TopicModelType.LDA_GIBBS:
                TopicModelType.LDA_GIBBS.value.seed_rng(value)
        super().__setattr__(name, value)

    def copy(self) -> "TopicModel":
        """
        Slightly shallow copy, last query state is not amended
        :return:
        """
        result = TopicModel()
        result.set_params(**result.get_params())
        result._model_state = _lda_cvb0.newModelFromExisting(self._model_state)
        result._last_query_state = self._last_query_state
        return result
    
    

    def fit_transform(self,  X: Union[np.ndarray, DataSet], y: np.ndarray = None, **kwargs) -> np.ndarray:
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
            X: Union[np.ndarray, DataSet],
            y: np.ndarray = None,
            **kwargs) -> "TopicModel":
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
        if type(X) is not DataSet:
            X = DataSet(words=X)

        module = TopicModelType[self.kind].value
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
        if type(X) is not DataSet:
            X = DataSet(words=X)

        logging.debug("Transforming data")
        module = TopicModelType[self.kind].value
        if self._model_state is None:
            raise ValueError("Untrained model")

        input_query = self.make_or_resume_query_state(X, kwargs.get('resume'))

        iters = kwargs.get('iterations') or self.query_iterations
        query_plan = self._new_train_plan(module, iters, **kwargs)

        self._model_state, inferred_query_state = module.query(X, self._model_state, input_query, query_plan)

        self.transform_metrics_ = None
        self._bound = math.nan
        self._set_or_clear_last_query_state(X, inferred_query_state, kwargs.get('persist_query_state'))

        ret_val = module.topicDists(inferred_query_state)
        logging.info(f"Obtained a {ret_val.shape} topic-assignments matrix after {iters} iterations")
        return ret_val

    def make_or_resume_query_state(self, X: DataSet, should_resume: bool):
        module = TopicModelType[self.kind].value
        if should_resume:
            if self._last_X is X:
                logging.info("Resuming from last query state")
                input_query = self._last_query_state
            else:
                raise ValueError("Attempting to resume with a different dataset to last fit/transform run")
        else:
            logging.info("Creating new query state")
            input_query = module.newQueryState(X, self._model_state, self.debug)
        return input_query

    @property
    def components_(self):
        module = TopicModelType[self.kind].value
        return module.wordDists(self._model_state)

    @property
    def query_state_(self):
        return self._last_query_state

    def score(self, X: DataSet, y: np.ndarray = None, y_query_state: object = None, method: Union[ScoreMethod, str] = None, **kwargs) -> float:
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
        module = TopicModelType[self.kind].value
        if type(X) is not DataSet:
            X = DataSet(words=X)
        if method is None:
            method = self.default_scoring_method
        elif type(method) is str:
            method = ScoreMethod.from_str(method)

        if method.is_doc_completion() and ((y is not None) or (y_query_state) is not None):
            logging.warning("Presuming X and y/y_query_state have been inferred appropriately for document completion")

        if (y is not None) and (y_query_state is not None):
            raise ValueError("Cannot specify both y and y_query_state at the same time")
        elif (y is None) and (y_query_state is None):
            if not method.is_doc_completion():
                logging.warning("No representation (y), or distribution (y_query_state), querying for y instead")
                y = self.transform(X)
            else:
                test_infer, test_eval = X.doc_completion_split()
                y = self.transform(test_infer)
                X = test_eval
        elif y is not None:
            if not method.is_point_estimate():
                raise ValueError("Only support point scoring methods for point estimates of y. "
                                 "Pass in the full query-state to get a Bayesian estimate")
            query_state = self.make_or_resume_query_state(X, should_resume=False)
            query_state_fields = dir(query_state)
            if "topicDists" in query_state_fields:
                query_state = query_state._replace(topicDists=(y * X.doc_lens[:, np.newaxis]).astype(self._model_state.dtype))
            elif "n_dk" in query_state_fields:
                n_dk = (y * X.doc_lens[:, np.newaxis]).astype(self._model_state.dtype)
                n_k = n_dk.sum(axis=0)
                query_state = query_state._replace(n_dk=n_dk, n_k=n_k)
            else:
                raise ValueError("Can't amend query-state")
        elif y_query_state is not None:
            query_state = y_query_state
            y = module.topicDists(y_query_state)

        if method.is_point_estimate():
            logging.info("Obtaining point estimate of log-likelihodd")
            log_prob = self.log_likelihood_score_point(X, y).sum()
        else:
            logging.info("Obtaining expected value (samples, variational bound) of log likelihood")
            log_prob = module.log_likelihood_expected(X, self._model_state, query_state).sum()

        if method.is_perplexity():
            logging.info("Converting log likelihood to perplexity")
            return module.perplexity_from_like(
                log_prob,
                X.word_count
            )
        else:
            return log_prob

    def log_likelihood_score_point(self, X: DataSet, y: np.ndarray) -> np.ndarray:
        """
        Return the log-likelihood for each individual document, according to the
        underlying model

        :param X: the input documents.
        :param y: the per-document topic weightings
        :return:  a log-likelihood per document, should be doc-count x 1
        """
        return TopicModel._log_likelihood_score_point(TopicModelType[self.kind],
                                                      X=X,
                                                      weightings=y,
                                                      components=self.components_)

    @staticmethod
    def _log_likelihood_score_point(model_type: TopicModelType,
                                    X: DataSet,
                                    weightings: np.ndarray,
                                    components: np.ndarray) -> np.ndarray:
        """
        For a mixture model, where weightings are θ and components are ϕ, the
        log likelihood is

        ln p(x|X) =~ ln p(x; θ, ϕ) = ln (Σ_k θ[k] Π_t x_t ^ ϕ_k)

        For an admixture model, where weightings are θ and components are ϕ, the
        log likelihood is

        ln p(x|X) =~ ln p(x; θ, ϕ) = ln (Π_n Π_t x_nt ^ (θ_k ϕ_kt))
                                   = Σ_n Σ_t x_nt ln (θ_k ϕ_kt)

        The type we use is determined by the topic-model type

        :param model_type: the type of model, specifically we're intested to see if
        this is a mixture or an admixture.
        :param X: the input documents.
        :param weightings: the per-document weightings, or a single per-corpus weighting
        as used by mixture models. Should be doc-count x topic-count
        :param components: the individual components. Should be topic-count x input-dimension
        :return: a log-likelihood per document, should be doc-count x 1
        """
        if model_type.is_admixture():
            words = X.words
            if words.dtype != components.dtype:
                words = words.astype(components.dtype)

            #       Returns X.words * np.log(weightings @ components)
            return sparseScalarProductOfSafeLnDot(words, weightings, components)
        elif model_type.is_mixture():
            lnWeights = safe_log(weightings)
            lnComps = safe_log(components)

            # Get the log-likelihoods for each component
            per_doc_log_likes = (X.words @ lnComps.T)
            if lnWeights.ndim == 1 or ((lnWeights.shape[0] == 1) and (X.words.shape[0] > 1)):
                per_doc_log_likes += lnWeights[np.newaxis, :]
            else:
                per_doc_log_likes += lnWeights

            # substract off a factor to avoid overflow/ underflow, then exponentiate to get likelioods
            # (this is the usual log-sum-exp formula used for multinomial likelihoods)
            max_log_like = per_doc_log_likes.max(axis=1)
            per_doc_log_likes -= max_log_like[:, np.newaxis]
            adj_per_doc_likes = np.exp(per_doc_log_likes, out=per_doc_log_likes)

            # sum to get overall likelihood across all topics, adding back in the factor we used
            # to control overflow.
            result = adj_per_doc_likes.sum(axis=1)
            result += max_log_like

            return result
        else:
            raise ValueError(f"Unknown model-type: {model_type}")



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



class WrappedSckitLda(LatentDirichletAllocation):

    def __init__(self,
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
                 default_scoring_method: ScoreMethod = ScoreMethod.LogLikelihoodPoint,
                 debug: bool = False,
                 seed: int = 0xC0FFEE) -> None:
        if batchSize == 0:
            super().__init__(
                n_components=n_components,
                doc_topic_prior=doc_topic_prior,
                topic_word_prior=topic_word_prior,
                learning_method='batch',
                max_iter=iterations,
                random_state=RandomState(seed),
                verbose=int(debug),
                perp_tol=perp_tolerance
            )
        else:
            super().__init__(
                n_components=n_components,
                doc_topic_prior=doc_topic_prior,
                topic_word_prior=topic_word_prior,
                learning_method='online',
                learning_decay=rate_b,
                learning_offsetfloat=rate_a,
                batch_size=batchSize,
                max_iter=iterations,
                random_state=RandomState(seed),
                verbose=int(debug),
                perp_tol=perp_tolerance
            )
        self.skip_init = False
        self.default_scoring_method = default_scoring_method

    def _init_latent_vars(self, n_features: int) -> None:
        if self.skip_init:
            return
        else:
            super()._init_latent_vars(n_features)

    def fit(self,
            X: Union[DataSet, np.ndarray],
            y: np.ndarray = None,
            **kwargs) -> "WrappedSckitLda":
        try:
            if kwargs.get('resume'):
                self.skip_init = True

            if type(X) is DataSet:
                X = X.words

            for k, v in kwargs.items():
                if k != 'resume':
                    logging.warning(f"Ignoring kwargs parameter {k}={v}")

            super().fit(X, y)
            return self
        finally:
            self.skip_init = False

    def fit_transform(self, X: Union[DataSet, np.ndarray], y: np.ndarray = None, **kwargs) -> np.ndarray:
        if type(X) is DataSet:
            X = X.words
        return super().fit_transform(X, y, **kwargs)

    def transform(self,
                  X: Union[DataSet, np.ndarray],
                  y: np.ndarray = None,
                  **kwargs) -> np.ndarray:
        if type(X) is DataSet:
            X = X.words
        if y is not None:
            raise ValueError("y is not allowed for wrapped scikit learn LDA")
        return super().transform(X)

    def score(self, X: Union[DataSet, np.ndarray], y: np.ndarray = None, y_query_state: object = None,
              method: Union[ScoreMethod, str] = None, **kwargs) -> float:
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
        if type(X) is not DataSet:
            X = DataSet(words=X)

        if method is None:
            method = self.default_scoring_method
        elif type(method) is str:
            method = ScoreMethod.from_str(method)

        if method.is_doc_completion() and ((y is not None) or (y_query_state) is not None):
            logging.warning("Presuming X and y/y_query_state have been inferred appropriately for document completion")

        if method not in [ScoreMethod.LogLikelihoodPoint, ScoreMethod.PerplexityPoint,
                          ScoreMethod.DocCompletionLogLikelihoodPoint, ScoreMethod.DocCompletionPerplexityPoint]:
            raise NotImplementedError(f"Method {method} not implemented")

        if method.is_doc_completion() and ((y is not None) or (y_query_state) is not None):
            logging.warning("Presuming X and y/y_query_state have been inferred appropriately for document completion")

        if (y is not None) and (y_query_state is not None):
            raise ValueError("Cannot specify both y and y_query_state at the same time")
        elif (y is None) and (y_query_state is None):
            if not method.is_doc_completion():
                logging.warning("No representation (y), or distribution (y_query_state), querying for y instead")
                y = self.transform(X)
            else:
                test_infer, test_eval = X.doc_completion_split()
                y = self.transform(test_infer)
                X = test_eval
        # elif y is not None:  The limitation to non-Bayesian estimates means we don't have to worry about this case
        elif y_query_state is not None:
            raise ValueError("No such thing as a query-state for Wrapped LDA")

        component_dists = self.components_ / self.components_.sum(axis=1)[:, np.newaxis]
        log_probs = sparseScalarProductOfSafeLnDot(X.words, y, component_dists).sum(axis=1)
        log_probs = np.squeeze(np.array(log_probs))
        log_prob  = log_probs.sum()

        if method.is_perplexity():
            from sidetopics.model.evals import perplexity_from_like

            logging.info("Converting log likelihood to perplexity")
            return perplexity_from_like(
                log_prob,
                X.word_count
            )
        else:
            return log_prob


class WrappedScikitHdp(HdpTransformer):

    def __init__(self,
                 dictionary: Dictionary,
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
                 default_scoring_method: ScoreMethod = ScoreMethod.LogLikelihoodPoint,
                 debug: bool = False,
                 expected_corpus_size: int = 10_000,
                 seed: int = 0xC0FFEE) -> None:

        if batchSize == 0:
            super().__init__(
                id2word=dictionary,
                T=n_components,
                K=n_components,
                max_chunks=iterations,
                chunksize=expected_corpus_size,
                kappa=forgetting_rate,
                tau=rate_delay,
                random_state=RandomState(seed),
            )
        else:
            super().__init__(
                id2word=dictionary,
                T=n_components,
                K=n_components,
                max_chunks=iterations,
                chunksize=batchSize,
                kappa=forgetting_rate,
                tau=rate_delay,
                random_state=RandomState(seed)
            )
        self.default_scoring_method = default_scoring_method

    @property
    def components_(self) -> np.ndarray:
        return self.gensim_model.m_lambda / self.gensim_model.m_lambda_sum[:, np.newaxis]

    def fit(self,
            X: Union[DataSet, np.ndarray],
            y: np.ndarray = None,
            **kwargs) -> "WrappedScikitHdp":
        if type(X) is DataSet:
            X = X.words
        if ssp.issparse(X):
            X = matutils.Sparse2Corpus(sparse=X, documents_columns=False)

        for k, v in kwargs.items():
            if k != 'resume':
                logging.warning(f"Ignoring kwargs parameter {k}={v}")

        if kwargs.get('resume'):
            super().partial_fit(X)
        else:
            super().fit(X, y)
        return self

    def fit_transform(self, X: Union[DataSet, np.ndarray], y: np.ndarray = None, **kwargs) -> np.ndarray:
        if type(X) is DataSet:
            X = X.words
        if ssp.issparse(X):
            X = matutils.Sparse2Corpus(sparse=X, documents_columns=False)
        return super().fit_transform(X, y, **kwargs)

    def transform(self,
                  X: Union[DataSet, np.ndarray],
                  y: np.ndarray = None,
                  **kwargs) -> np.ndarray:
        if type(X) is DataSet:
            X = X.words
        if ssp.issparse(X):
            X = matutils.Sparse2Corpus(sparse=X, documents_columns=False)
        if y is not None:
            raise ValueError("y is not allowed for wrapped scikit learn LDA")
        return super().transform(X)

    def score(self, X: Union[DataSet, np.ndarray], y: np.ndarray = None, y_query_state: object = None,
              method: Union[ScoreMethod, str] = None, **kwargs) -> float:
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
        if type(X) is not DataSet:
            X = DataSet(words=X)

        if method is None:
            method = self.default_scoring_method
        elif type(method) is str:
            method = ScoreMethod.from_str(method)

        if method not in [ScoreMethod.LogLikelihoodPoint, ScoreMethod.PerplexityPoint,
                          ScoreMethod.DocCompletionLogLikelihoodPoint, ScoreMethod.DocCompletionPerplexityPoint]:
            raise NotImplementedError(f"Method {method} not implemented")

        if (y is not None) and (y_query_state is not None):
            raise ValueError("Cannot specify both y and y_query_state at the same time")
        elif (y is None) and (y_query_state is None):
            if not method.is_doc_completion():
                logging.warning("No representation (y), or distribution (y_query_state), querying for y instead")
                y = self.transform(X)
            else:
                test_infer, test_eval = X.doc_completion_split()
                y = self.transform(test_infer)
                X = test_eval
        # elif y is not None:  The limitation to non-Bayesian estimates means we don't have to worry about this case
        elif y_query_state is not None:
            raise ValueError("No such thing as a query-state for Wrapped LDA")

        component_dists = self.components_ / self.components_.sum(axis=1)[:, np.newaxis]
        if y.dtype != X.words.dtype:
            y =  y.astype(X.words.dtype)
        if component_dists.dtype != X.words.dtype:
            component_dists =  component_dists.astype(X.words.dtype)

        log_probs = sparseScalarProductOfSafeLnDot(X.words, y, component_dists).sum(axis=1)
        log_probs = np.squeeze(np.array(log_probs))
        log_prob  = log_probs.sum()

        if method.is_perplexity():
            from sidetopics.model.evals import perplexity_from_like

            logging.info("Converting log likelihood to perplexity")
            return perplexity_from_like(
                log_prob,
                X.word_count
            )
        else:
            return log_prob
