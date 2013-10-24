#!/usr/bin/python
# -*- coding: utf-8 -*- 
'''
A collection of functions that perform various log / exp related calculations
without over or under flowing on numpy buffers. 


Created on 11 Jul 2013

@author: bryanfeeney
'''

import numpy as np

# TODO This works, but how and why does it work?
def safe_log_one_plus_exp_of (x):
    '''
    Avoids overflow and underflow when calculating log(1+exp(x)).
    Taken from Guillaume Bouchard's CTM code
    '''
    LOWER = -17
    UPPER =  33
    
    out = np.ndarray(x.shape)
    out[x <= LOWER] = np.exp(x[x <= LOWER])
    
    selection      = np.all([x > LOWER, x <= 0], axis=0)
    out[selection] = np.log (1 + np.exp(x[selection]))
    
    selection      = np.all([x > 0, x <= UPPER], axis=0)
    out[selection] = x[selection] + np.log (1 + np.exp(-x[selection]))
        
    out[x > UPPER] = x[x > UPPER]
    
    return out
    
    

# TODO Can this be done in a single pass like Guillaume's CTM?
# TODO Need to adjust for 32-bit floats
# TODO Returns -inf when x = 0, but by rights should return log(2) ~= 0.693
def _very_safe_log_one_plus_exp_of (x):
    '''
    Avoids overflow and underflow when calculating log(1+exp(x)). A good description of 
    considerations for this specific function is given at
    
    http://www.johndcook.com/blog/2008/04/16/overflow-and-loss-of-precision/
    
    With code samples given by the same author in this document.
    
    http://www.codeproject.com/Articles/25294/Avoiding-Overflow-Underflow-and-Loss-of-Precision
    
    Params
    x - A numpy buffer, assumes to have double-precision floating point numbers
        (THIS BREAKS FOR SINGLE PRECISION FLOATING POINT!)
    
    Returns
    A numpy buffer of the same size, such that each element is log(1+exp(x))
    '''
    DBL_EPSILON     = 2.2204460492503131e-16
    LOG_DBL_EPSILON = np.log(DBL_EPSILON);
    LOG_ONE_QUARTER = np.log(0.25);

    out = np.ndarray(x.shape)
    

    # Feasible region
    out[x > LOG_ONE_QUARTER] = np.log(1. + np.exp(x[x > LOG_ONE_QUARTER]))
    
    # log(exp(x) + 1) == x to machine precision
    out[x > -LOG_DBL_EPSILON] = x[x > -LOG_DBL_EPSILON]
    
    # For smaller than this we need to work around the problem of 
    # adding a miniscule argument to 1, which will be a relatively huge
    # number
    out[x <= LOG_ONE_QUARTER] = safe_log_one_plus (np.exp(x[x <= LOG_ONE_QUARTER]))
    
    return out

def safe_log_one_plus(x):
    '''
    Calculates log(1+x), without overflowing.
        
    Transliterated from code provided with a description in this document.
    
    http://www.codeproject.com/Articles/25294/Avoiding-Overflow-Underflow-and-Loss-of-Precision
    
    Params
    x - A numpy buffer, assumes to have double-prescision floating point numbers
        (THIS BREAKS FOR FLOATING POINT!)
    
    Returns
    A numpy buffer of the same size, such that each element is log(1+x)
    '''
    
    out = np.ndarray(x.shape)
    out[x > 0.375] = np.log(1. + x[x > 0.375])

    # For smaller arguments we use a rational approximation
    # to the function log(1+x) to avoid the loss of precision
    # that would occur if we simply added 1 to x then took the log.

    p1 =  -0.129418923021993e+01;
    p2 =   0.405303492862024e+00;
    p3 =  -0.178874546012214e-01;
    
    q1 =  -0.162752256355323e+01;
    q2 =   0.747811014037616e+00;
    q3 =  -0.845104217945565e-01;

    buf = x[x <= 0.375]
    
    t = buf /(buf + 2.0);
    t2 = t*t;
    w = (((p3*t2 + p2)*t2 + p1)*t2 + 1.0)/(((q3*t2 + q2)*t2 + q1)*t2 + 1.0);
    
    out[x <= 0.375] = 2.0*t*w
    
    return out

# TODO: How slow is this...
def safe_x_log_x(x):
    '''
    An implementation of x * log(x), applied to arrays, which substitutes tiny numbers
    for values of zero in those arrays, so that the standard numpy log function
    will work
    '''
    
    almostZero = 1E-35 if x.dtype == np.float32 else 1E-300
    
    log_x  = np.ndarray(x.shape)
    log_x.fill(np.log(almostZero))
    
    log_x[x>0]  = np.log(x[x>0])
    return x * log_x


def safe_log (x, out = None):
    '''
    An implementation of log, applied to arrays, which substitutes tiny numbers
    for values of zero in those arrays, so that the standard numpy log function
    will work
    '''
    if out is None:
        out = np.ndarray(x.shape)
        
    almostZero = 1E-35 if x.dtype == np.float32 else 1E-300
    out.fill(np.log(almostZero))
    
    out[x>0] = np.log(x[x>0])
    return out
