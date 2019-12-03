import sys
import pathlib
import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse as ssp
import matplotlib.pyplot as plt
import seaborn as sns
from typing import NamedTuple, Any, List, Dict, Set, Union
import pickle as 
import gzip
import pandas as pd
import logging

DATASET_DIR = pathlib.Path('/') / 'Volumes' / 'DatasetSSD'

class LabelledMatrix:
    """
    A matrix with a set of labels for every column
    """
    values: Union[ssp.csr_matrix, np.ndarray]
    labels: Any = None
        
    def __init__(self, values, labels=None):
        self.values = values
        self.labels = labels
        
        if self.values is not None and self.labels is not None:
            assert(self.values.shape[1] == len(self.labels)), \
                f"Matrix has shape {self.values.shape} but labels has length {len(self.labels)}"
            
    def __str__(self):
        l = "" if self.labels is None else "(labelled)"
        return f"[{self.values.shape}]{l}"
    
    
class RawData(NamedTuple):
    """
    A dataset, being a collection of labelled matrices and a single set of row-labels
    shared between all of them.
    """
    words: LabelledMatrix
    feats: LabelledMatrix = None
    cites: LabelledMatrix = None
    authors: LabelledMatrix = None
    categories: LabelledMatrix = None
    row_labels: List[str] = None
        
    def __str__(self):
        return f"RawData(words{self.words}, feats{self.feats}, cites{self.cites}, authors{self.authors}, " \
               f"categories{self.categories}, row_labels[{'' if self.row_labels is None else len(self.row_labels)}]"
    

