#!/usr/bin/env python
import pandas as pd
import numpy as np

def get_probs(seqs,starts,width,pseudo):
    """TODO: Docstring for get_probs.

    Parameters
    ----------
    seqs : list
        List of sequences
    starts : list
        List of starts for the motif in the corresponding sequence
    width : int
        Length of motif
    pseudo: int
        pseudocount to use

    Returns
    -------
    probability matrix in the form of:

            0    1    2 .... W
        A  0.0  0.0
        C  0.0  0.0
        G  0.0  0.0
        T  0.0  0.0

        Where 0 is background, 1 is the first column of the motif

    """

    _n = pd.DataFrame(data=np.zeros((4,width+1)),index=['A','C','G','T'],columns=list(range(width+1)))
    _p = pd.DataFrame(data=np.zeros((4,width+1)),index=['A','C','G','T'],columns=list(range(width+1)))

    for i, sequence in enumerate(seqs):
        for j, base in enumerate(sequence):
            # In motif
            if j >= starts[i] and j < starts[i] + width:
                k = j - starts[i]
                _n[base,k] += 1
            else:
                _n[base,0] += 1
    for base in range(_p.shape[0]):
        for position in range(_p.shape[1]):
            if positition > 0:
                _p[base,position] = np.log((_n[base,position] + pseudo)/(len(sequences) - 1 + 4*pseudo))
            else:
                _p[base,position] = np.log(
                    (_n[base,position] + pseudo)\((len(sequences) - 1)(len(sequences[0]) - width) + 4*pseudo))
    return _p

def likelihood(seqs,starts):
    """TODO: Docstring for likelihood.

    Parameters
    ----------
    seqs : TODO
    starts : TODO

    Returns
    -------
    TODO

    """
    _p = get_probs(seq, seqs, starts, width, pseudo)
    end = len(seq) - width + 1

    LRs = [0]*(end + 1)
    tot = float('-inf')
    for j in range(end):
       log_prob_motif = np.sum([_p.loc[base,k+1] for k,base in enumerate(seq[j:j+width])])
       log_prob_background = np.sum([_p.loc[base,0] for base in seq[j:j+width]])
       LR = log_prob_motif - log_prob_background
       LRs[j] = LR
       tot = LR + np.log(1 + np.exp(tot - LR))
    LRs /= tot
    #Randomly select?
    np.


