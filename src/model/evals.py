'''
Created on 20 Apr 2015

@author: bryanfeeney
'''

import numpy as np

def perplexity(log_likely, token_count):
    return np.exp(-log_likely / token_count)

def meanAveragePrec():
    '''
    Calculates the average (mean) of average-precisions. See averagePrec for
    a description of the latter. 
    '''
    
    
def averagePrec(expectedLinks, givenLinks):
    '''
    Calculates the average precision for a document. For each link in the
    expected set, we find it's position in the given set, and then calcualte
    the precision at M
    
    This metric penalises cases where there is a lot of irrelevant links
    ahead of and/or between relevant links.
    '''
    
    
    
def precAtM(m, expectedLinks, givenLinks):
    '''
    The evaluates the precision at M for a document, which is to say, from all
    the links generated for a document, we take the first M, in order of
    predicted probabilitiy, and evaluate the precision of that set, i.e. the
    proportion of the expected links found in the given links
    
    m - how many of the first given links to consider
    expectedLinks - the links we expect to find
    givenLinks    - the links we actually found
    '''
    
    recv = set(givenLinks[:m])
    expt = set(expectedLinks)
    
    return len (recv.intersection(expt)) / len(expt)
    
    