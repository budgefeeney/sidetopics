

# 1 A fixture for dummy data that LDA should fit

# 2 Test of CVB 0

from typing import NamedTuple
import numpy as np
import numpy.testing as nptest
import numpy.random as rd
import unittest

from sidetopics.model import DataSet
from sidetopics.model.sklearn import *
from sidetopics.model.sklearn.lda_cvb import TopicModelType, WrappedSckitLda, WrappedScikitHdp, ScoreMethod
from sidetopics.model.evals import perplexity_from_like

AVG_DOC_LEN = 250
DOC_COUNT = 100
TRUE_TOPIC_COUNT = 5
WORDS_PER_TOPIC = 4  # We actually construct a clustering problem just to test, each topic generates WORDS_PER_TOPIC
                     # terms only, and none others, so that the total vocab size is TRUE_TOPIC_COUNT * WORDS_PER_TOPIC

class TopicModelTestSample:
    components: np.ndarray
    assignments: np.ndarray
    lengths: np.ndarray
    sample_documents: np.ndarray

    def __init__(self, components: np.ndarray, assignments: np.ndarray, lengths: np.ndarray):
        """
        For D documents with T words using K topics
        :param components:  K x T matrix of components
        :param assignments: D x K matrix of topic assignments
        :param lengths:  D x 1 vector of lengths
        :return:
        """
        self.components = components
        self.assignments = assignments
        self.lengths = lengths

        self.sample_documents = np.floor((assignments @ components) * lengths[:, np.newaxis])

    def as_dataset(self, **kwargs) -> DataSet:
        """
        Converts to a dataset, words only. Passes the extra parameters along to
        the DataSet constructor if any
        """
        return DataSet(words=self.sample_documents, **kwargs)

    @property
    def n_components(self):
        return self.components.shape[0]

    @staticmethod
    def new_fixed(seed: int = None):
        if seed is not None:
            rd.seed(seed)

        components = np.array([
            [0] * WORDS_PER_TOPIC*0 +  [1] * WORDS_PER_TOPIC + [0] * 4*WORDS_PER_TOPIC,
            [0] * WORDS_PER_TOPIC*1 + [1] * WORDS_PER_TOPIC + [0] * 3*WORDS_PER_TOPIC,
            [0] * WORDS_PER_TOPIC*2 + [1] * WORDS_PER_TOPIC + [0] * 2*WORDS_PER_TOPIC,
            [0] * WORDS_PER_TOPIC*3 + [1] * WORDS_PER_TOPIC + [0] * 1*WORDS_PER_TOPIC,
            [0] * WORDS_PER_TOPIC*4 + [1] * WORDS_PER_TOPIC + [0] * 0*WORDS_PER_TOPIC
        ])

        assignments = rd.dirichlet(alpha=[0.1] * TRUE_TOPIC_COUNT, size=DOC_COUNT)
        lens = rd.poisson(AVG_DOC_LEN, size=DOC_COUNT)
        return TopicModelTestSample(components=components,
                                    assignments=assignments,
                                    lengths=lens)


class ValidationRange(NamedTuple):
    min_excl: float
    max_excl: float

    def in_range(self, val: float) -> bool:
        return self.min_excl < val < self.max_excl

    def assert_in_range(self, test: unittest.TestCase, **kwargs):
        assert len(kwargs) == 1, "Only only key-word argument should be provided"
        name = next(iter(kwargs.keys()))
        val = kwargs[name]
        test.assertTrue(self.in_range(val), f"Value of {name}, {val:.1f}, not in exclusive "
                                            f"range {self.min_excl}..{self.max_excl}")



LDA_PERP_RANGE = ValidationRange(min_excl=5.5, max_excl=7.5)
LDA_LN_LIKE_RANGE = ValidationRange(min_excl=-200000, max_excl=-180000)

MOM_PERP_RANGE = ValidationRange(min_excl=8, max_excl=11)
MOM_LN_LIKE_RANGE = ValidationRange(min_excl=-230000, max_excl=-210000)

class SklearnLdaCvbTest(unittest.TestCase):
    # Add a test to ensure that repeated calls to transform have the same effect (i.e. we're not training by accident)

    skip_tests_on_expected_bounds: bool = True

    def test_lda_cvb0_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_CVB0, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset)

        model = TopicModel(kind=TopicModelType.LDA_CVB0, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments_2 = model.fit_transform(dataset, persist_query_state=True)
        qs = model._last_query_state
        np.testing.assert_array_almost_equal(assignments, assignments_2, decimal=3,
                                             err_msg="Failed to respect initial seed")

        score_0p = model.score(dataset, y=assignments, method=ScoreMethod.LogLikelihoodPoint)
        perp_0p = model.score(dataset, y=assignments, method=ScoreMethod.PerplexityPoint)
        LDA_PERP_RANGE.assert_in_range(self, point_perplexity=perp_0p)
        LDA_LN_LIKE_RANGE.assert_in_range(self, point_log_likely=score_0p)

        if not self.skip_tests_on_expected_bounds:
            score_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.LogLikelihoodBoundOrSampled)
            perp_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.PerplexityBoundOrSampled)
            LDA_PERP_RANGE.assert_in_range(self, expected_perplexity=perp_0e)
            LDA_LN_LIKE_RANGE.assert_in_range(self, expected_log_likely=score_0e)

            self.assertTrue(perp_0e > perp_0p, f"Expected perplexity {perp_0e} is not greater than {perp_0p}")


    def test_lda_cvb0_resume_simple_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_CVB0, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset, iterations=100)

        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_CVB0, n_components=testcase.n_components, seed=0xC0FFEE)
        _ = model.fit_transform(dataset, iterations=90, persist_query_state=True)
        assignments_from_resume = model.fit_transform(dataset, iterations=10, resume=True)

        np.testing.assert_array_almost_equal(assignments, assignments_from_resume, decimal=3)


    def test_mom_vb_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.MOM_VB, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset)

        model = TopicModel(kind=TopicModelType.MOM_VB, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments_2 = model.fit_transform(dataset, persist_query_state=True)
        qs = model._last_query_state
        np.testing.assert_array_almost_equal(assignments, assignments_2, decimal=3,
                                             err_msg="Failed to respect initial seed")

        score_0p = model.score(dataset, y=assignments, method=ScoreMethod.LogLikelihoodPoint)
        perp_0p = model.score(dataset, y=assignments, method=ScoreMethod.PerplexityPoint)
        MOM_PERP_RANGE.assert_in_range(self, point_perplexity=perp_0p)
        MOM_LN_LIKE_RANGE.assert_in_range(self, point_log_likely=score_0p)

        if not self.skip_tests_on_expected_bounds:
            score_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.LogLikelihoodBoundOrSampled)
            perp_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.PerplexityBoundOrSampled)
            MOM_PERP_RANGE.assert_in_range(self, expected_perplexity=perp_0e)
            MOM_LN_LIKE_RANGE.assert_in_range(self, expected_log_likely=score_0e)

            self.assertTrue(perp_0e > perp_0p, f"Expected perplexity {perp_0e} is not greater than {perp_0p}")


    def test_mom_vb_resume_simple_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.MOM_VB, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset, iterations=100)

        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.MOM_VB, n_components=testcase.n_components, seed=0xC0FFEE)
        _ = model.fit_transform(dataset, iterations=90, persist_query_state=True)
        assignments_from_resume = model.fit_transform(dataset, iterations=10, resume=True)

        np.testing.assert_array_almost_equal(assignments, assignments_from_resume, decimal=3)



    def test_mom_gibbs_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.MOM_GIBBS, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset)

        model = TopicModel(kind=TopicModelType.MOM_GIBBS, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments_2 = model.fit_transform(dataset, persist_query_state=True)
        qs = model._last_query_state
        np.testing.assert_array_almost_equal(assignments, assignments_2, decimal=3,
                                             err_msg="Failed to respect initial seed")

        score_0p = model.score(dataset, y=assignments, method=ScoreMethod.LogLikelihoodPoint)
        perp_0p = model.score(dataset, y=assignments, method=ScoreMethod.PerplexityPoint)
        MOM_PERP_RANGE.assert_in_range(self, point_perplexity=perp_0p)
        MOM_LN_LIKE_RANGE.assert_in_range(self, point_log_likely=score_0p)

        if not self.skip_tests_on_expected_bounds:
            score_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.LogLikelihoodBoundOrSampled)
            perp_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.PerplexityBoundOrSampled)
            MOM_PERP_RANGE.assert_in_range(self, expected_perplexity=perp_0e)
            MOM_LN_LIKE_RANGE.assert_in_range(self, expected_log_likely=score_0e)

            self.assertTrue(perp_0e > perp_0p, f"Expected perplexity {perp_0e} is not greater than {perp_0p}")


    def test_mom_gibbs_resume_simple_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.MOM_GIBBS, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset, iterations=100)

        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.MOM_GIBBS, n_components=testcase.n_components, seed=0xC0FFEE)
        _ = model.fit_transform(dataset, iterations=90, persist_query_state=True)
        assignments_from_resume = model.fit_transform(dataset, iterations=10, resume=True)

        np.testing.assert_array_almost_equal(assignments, assignments_from_resume, decimal=3)


    def test_lda_cvb_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_CVB, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset)

        model = TopicModel(kind=TopicModelType.LDA_CVB, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments_2 = model.fit_transform(dataset, persist_query_state=True)
        qs = model._last_query_state
        np.testing.assert_array_almost_equal(assignments, assignments_2, decimal=3,
                                             err_msg="Failed to respect initial seed")

        score_0p = model.score(dataset, y=assignments, method=ScoreMethod.LogLikelihoodPoint)
        perp_0p = model.score(dataset, y=assignments, method=ScoreMethod.PerplexityPoint)
        LDA_PERP_RANGE.assert_in_range(self, point_perplexity=perp_0p)
        LDA_LN_LIKE_RANGE.assert_in_range(self, point_log_likely=score_0p)

        if not self.skip_tests_on_expected_bounds:
            score_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.LogLikelihoodBoundOrSampled)
            perp_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.PerplexityBoundOrSampled)
            LDA_PERP_RANGE.assert_in_range(self, expected_perplexity=perp_0e)
            LDA_LN_LIKE_RANGE.assert_in_range(self, expected_log_likely=score_0e)

            self.assertTrue(perp_0e > perp_0p, f"Expected perplexity {perp_0e} is not greater than {perp_0p}")


    def test_lda_cvb_resume_simple_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_CVB, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset, iterations=100)

        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_CVB, n_components=testcase.n_components, seed=0xC0FFEE)
        _ = model.fit_transform(dataset, iterations=90, persist_query_state=True)
        assignments_from_resume = model.fit_transform(dataset, iterations=10, resume=True)

        np.testing.assert_array_almost_equal(assignments, assignments_from_resume, decimal=3)


    def test_wrapped_lda_vb_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = WrappedSckitLda(n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset)

        model = WrappedSckitLda(n_components=testcase.n_components, seed=0xC0FFEE)
        assignments_2 = model.fit_transform(dataset, persist_query_state=True)
        np.testing.assert_array_almost_equal(assignments, assignments_2, decimal=3,
                                             err_msg="Failed to respect initial seed")

        score_0p = model.score(dataset, y=assignments, method=ScoreMethod.LogLikelihoodPoint)
        perp_0p = model.score(dataset, y=assignments, method=ScoreMethod.PerplexityPoint)
        LDA_PERP_RANGE.assert_in_range(self, point_perplexity=perp_0p)
        LDA_LN_LIKE_RANGE.assert_in_range(self, point_log_likely=score_0p)

        if not self.skip_tests_on_expected_bounds:
            raise NotImplementedError()


    def test_wrapped_lda_vb_resume_simple_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = WrappedSckitLda(n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset, iterations=100)

        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = WrappedSckitLda(n_components=testcase.n_components, seed=0xC0FFEE)
        _ = model.fit_transform(dataset, iterations=90, persist_query_state=True)
        assignments_from_resume = model.fit_transform(dataset, iterations=10, resume=True)

        np.testing.assert_array_almost_equal(assignments, assignments_from_resume, decimal=2)

    def test_gensim_dict_generation_for_hdp(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)
        dictionary = dataset.gensim_dictionary()
        print(str(dictionary))

        self.assertEqual(
            [(i, str(i)) for i in range(int(dataset.words.shape[1]))],
            sorted(dictionary.items())
        )

    def test_wrapped_hdp_vb_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)
        dict =  dataset.gensim_dictionary()

        model = WrappedScikitHdp(n_components=testcase.n_components,
                                 expected_corpus_size=dataset.doc_count,
                                 dictionary=dict,
                                 seed=0xC0FFEE)
        assignments = model.fit_transform(dataset)

        model = WrappedScikitHdp(n_components=testcase.n_components,
                                 expected_corpus_size=dataset.doc_count,
                                 dictionary=dict,
                                 seed=0xC0FFEE)
        assignments_2 = model.fit_transform(dataset, persist_query_state=True)
        np.testing.assert_array_almost_equal(assignments, assignments_2, decimal=3,
                                             err_msg="Failed to respect initial seed")

        score_0p = model.score(dataset, y=assignments, method=ScoreMethod.LogLikelihoodPoint)
        perp_0p = model.score(dataset, y=assignments, method=ScoreMethod.PerplexityPoint)
        LDA_PERP_RANGE.assert_in_range(self, point_perplexity=perp_0p)
        LDA_LN_LIKE_RANGE.assert_in_range(self, point_log_likely=score_0p)

        if not self.skip_tests_on_expected_bounds:
            raise NotImplementedError()


    def test_wrapped_hdp_vb_resume_simple_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)
        dict =  dataset.gensim_dictionary()

        model = WrappedScikitHdp(n_components=testcase.n_components,
                                 expected_corpus_size=dataset.doc_count,
                                 dictionary=dict, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset, iterations=100)

        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = WrappedScikitHdp(n_components=testcase.n_components,
                                 expected_corpus_size=dataset.doc_count,
                                 dictionary=dict, seed=0xC0FFEE)
        _ = model.fit_transform(dataset, iterations=90, persist_query_state=True)
        assignments_from_resume = model.fit_transform(dataset, iterations=10, resume=True)

        np.testing.assert_array_almost_equal(assignments, assignments_from_resume, decimal=2)


    def test_lda_vb_python_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_VB_PYTHON_IMPL, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset)

        model = TopicModel(kind=TopicModelType.LDA_VB_PYTHON_IMPL, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments_2 = model.fit_transform(dataset, persist_query_state=True)
        qs = model._last_query_state
        np.testing.assert_array_almost_equal(assignments, assignments_2, decimal=3,
                                             err_msg="Failed to respect initial seed")

        score_0p = model.score(dataset, y=assignments, method=ScoreMethod.LogLikelihoodPoint)
        perp_0p = model.score(dataset, y=assignments, method=ScoreMethod.PerplexityPoint)
        LDA_PERP_RANGE.assert_in_range(self, point_perplexity=perp_0p)
        LDA_LN_LIKE_RANGE.assert_in_range(self, point_log_likely=score_0p)

        if not self.skip_tests_on_expected_bounds:
            score_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.LogLikelihoodBoundOrSampled)
            perp_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.PerplexityBoundOrSampled)
            LDA_PERP_RANGE.assert_in_range(self, expected_perplexity=perp_0e)
            LDA_LN_LIKE_RANGE.assert_in_range(self, expected_log_likely=score_0e)

            self.assertTrue(perp_0e > perp_0p, f"Expected perplexity {perp_0e} is not greater than {perp_0p}")


    def test_lda_vb_python_resume_simple_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_VB_PYTHON_IMPL, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset, iterations=100)

        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_VB_PYTHON_IMPL, n_components=testcase.n_components, seed=0xC0FFEE)
        _ = model.fit_transform(dataset, iterations=90, persist_query_state=True)
        assignments_from_resume = model.fit_transform(dataset, iterations=10, resume=True)

        np.testing.assert_array_almost_equal(assignments, assignments_from_resume, decimal=3)


    def test_lda_gibbs_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_VB_PYTHON_IMPL, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset)

        model = TopicModel(kind=TopicModelType.LDA_VB_PYTHON_IMPL, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments_2 = model.fit_transform(dataset, persist_query_state=True)
        qs = model._last_query_state
        np.testing.assert_array_almost_equal(assignments, assignments_2, decimal=3,
                                             err_msg="Failed to respect initial seed")

        score_0p = model.score(dataset, y=assignments, method=ScoreMethod.LogLikelihoodPoint)
        perp_0p = model.score(dataset, y=assignments, method=ScoreMethod.PerplexityPoint)
        LDA_PERP_RANGE.assert_in_range(self, point_perplexity=perp_0p)
        LDA_LN_LIKE_RANGE.assert_in_range(self, point_log_likely=score_0p)

        if not self.skip_tests_on_expected_bounds:
            score_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.LogLikelihoodBoundOrSampled)
            perp_0e = model.score(dataset, y_query_state=qs, method=ScoreMethod.PerplexityBoundOrSampled)
            LDA_PERP_RANGE.assert_in_range(self, expected_perplexity=perp_0e)
            LDA_LN_LIKE_RANGE.assert_in_range(self, expected_log_likely=score_0e)

            self.assertTrue(perp_0e > perp_0p, f"Expected perplexity {perp_0e} is not greater than {perp_0p}")


    def test_lda_gibbs_resume_simple_data(self):
        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_VB_PYTHON_IMPL, n_components=testcase.n_components, seed=0xC0FFEE)
        assignments = model.fit_transform(dataset, iterations=100)

        testcase = TopicModelTestSample.new_fixed(seed=0xBADB055)
        dataset = testcase.as_dataset(debug=True)

        model = TopicModel(kind=TopicModelType.LDA_VB_PYTHON_IMPL, n_components=testcase.n_components, seed=0xC0FFEE)
        _ = model.fit_transform(dataset, iterations=90, persist_query_state=True)
        assignments_from_resume = model.fit_transform(dataset, iterations=10, resume=True)

        np.testing.assert_array_almost_equal(assignments, assignments_from_resume, decimal=3)




    # Aim of the work is to compare MoM with LDA
    # To compare LDA Gibbs with LDA VB with LDA SVB
    # To compare LDA VB with HDP
    # To look into hyper learning with LDA

