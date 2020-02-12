#!/usr/bin/env python
# Written by: Nathan Richman <nate@nrichman.dev>

# For BMI 776 (Advanced Bioinformatics) at UW-Madison
# Spring 2020

# Part of homework 1, this file contains functions to output heights of characters in an
#   information content logo



import pandas as pd
import numpy as np

def _information_content(model):
    """Determines information content (IC) for a vector of distributions

    Parameters
    ----------
    model : array-like
        Array which holds parameters for a sequence motif model in the form of:
                    0    1    2 .... W
                A  0.25  0.2  0.1
                C  0.25  0.1  0.2
                G  0.25  0.2  0.1
                T  0.25  0.4  0.4

    Returns
    -------
    IC: 1D-array of IC values for each input distribution
    """
    max_entropy = np.log2(model.shape[0])
    IC = np.zeros((model.shape[1])) + max_entropy
    for dist in range(model.shape[1]):
        for entry in range(model.shape[0]):
            IC[dist] += np.log2(model.iloc[entry,dist])*model.iloc[entry,dist]
    return IC

def seq_logo(model):
    """Creates a Sequence logo using weblogo for a motif with given parameters

    Parameters
    ----------
    model : array-like
        Array which holds parameters for a sequence motif model in the form of:
                    0    1    2 .... W
                A  0.25  0.2  0.1
                C  0.25  0.1  0.2
                G  0.25  0.2  0.1
                T  0.25  0.4  0.4

    Returns
    -------
    Array like model but multiplied by the ICs for that distribution.  Each element i,j gives the height of element j at position i.

    """
    ICs = _information_content(model)
    for dist in range(model.shape[1]):
        model.iloc[:,dist] = model.iloc[:,dist]*ICs[dist]
    return model





