#!/usr/bin/env python
# Written by: Nathan Richman <nate@nrichman.dev>
# For BMI 776 (Advanced Bioinformatics) at UW-Madison
# Spring 2020

# Part of homework 1, this file contains classes and a main function to run the MEME algorithm
#   with an OOPS model and exhaustive subsequence method as a starting point.

# Framework code with a main function and the Argparser was provided.

import argparse, os, sys
import numpy as np
import pandas as pd
import collections
import seq_logo
import matplotlib.pyplot as plt
# You can choose to write classes in other python files
# and import them here.

class MEME(object):

    """Class for the EM algorithm"""


    def __init__(self, sequences, width, pseudocount):
        """TODO: to be defined.

        Parameters
        ----------
        sequences : array-like
            Array of sequences
        width: int
            Width of sequence to look for
        pseudocount: int
            Pseudocount to use when running the M-step

        """
        self._sequences = sequences
        if width > (len(sequences[0]) + 1)/2:
            raise ValueError("Width must be less than (len(sequence) + 1)/2")
        self._width = width
        self._pseudo = pseudocount
        # Number of starting positions, L - W + 1
        self._starting_pos = len(sequences[0]) - width + 1
        # Probability matrix, first column is background, starting at position 1 is the first position on the motif, initialize to all 0's
        # TODO check dimension on p, additionally check indexing on n_ck, p_ck p_ck and n_ck are 1-indexed since 0 is background, the rest is 0-indexed
        self._P = pd.DataFrame(data=np.zeros((4,width+1)),index=['A','C','G','T'],columns=list(range(width + 1)))
        # Matrix of starting position probabilities for each sequence, initialize to 0
        self._Z = np.zeros((len(sequences),self._starting_pos))

    @property
    def p(self):
        """Getter for _p

        Returns
        --------
        p_matrix

        """
        return self._P
    @property
    def z(self):
        """Getter for _z

        Returns
        --------
        z_matrix

        """
        return self._Z

    def _normalize(self,to_normalize):
        """_normalize: Helper function that normalizes a distribution in log-space so that all probabilities in the distribution sum to 1 in real space

        Parameters
        ----------
        to_normalize : 1D array
            Distribution in log-space to normalize

        Returns
        -------
        Normalized distribution in real space

        """

        # https://stats.stackexchange.com/questions/6616/converting-normalizing-very-small-likelihood-values-to-probability

        # For a precision of \eta = 10^-d for d digits of precision with n likelihoods
        #   1. Subtract the maximum likelihood from all logs.
        #   2. Throw away values less than the logarithm of \eta/n
        #
        # \eta = 10^-16 for 16 decimals
        # ln(\eta / n) = ln(\eta) - ln(n)
        thresh = np.log(10**-16) - np.log(len(to_normalize))
        my_max = np.max(to_normalize)
        # If the value is too small set it to float("-inf") which will successfully evaluat to 0 through the exp function
        vals = [np.exp(a -  my_max) if (a - my_max) > thresh else float("-inf") for a in to_normalize ]
        vals = vals/np.sum(vals)
        return vals

    def test(self,seq):
        counts = collections.Counter(''.join(self._sequences))
        p_matrix = pd.DataFrame(data=np.zeros_like(self._P),index=['A','C','G','T'],columns=list(range(self._width + 1)))
        ############ Populate p_c_k, k > 0
        # Go through bases, and set probability of the found character at that position equal to 2/5, and the others to 1/5
        counts_seq = collections.Counter(seq)
        for i,base in enumerate(seq):
            for letter in p_matrix.index:
                if letter == base:
                    p_matrix.loc[base,i+1] = (self._pseudo + 1)/(self._pseudo*4 + 1)
                else:

                    p_matrix.loc[letter,i+1] = (self._pseudo)/(self._pseudo*4 + 1)

        ############ Populate p_c_k, k = 0
        n_0 = {}
        for base in ['A','C','G','T']:
            # n_c_0 = total # of c's in dataset - count of c's in motif
            n_0[base] = (counts[base] - counts_seq[base])
        my_sum = sum(n_0.values())
        for i,base in enumerate(['A','C','G','T']):
            # p_matrix_0 = (n_c_0 + pseudocount)/sum n_b_0 + 4*pseudocount
            p_matrix.iloc[i,0] = (n_0[base] + self._pseudo)/(my_sum + self._pseudo*4)

        #TODO: Find what to do with background for now set all to .25
        # p_matrix.loc[:,0] = .25
        log_likelihood, p_matrix, z_matrix = self._run_em(p_matrix)
        self._P = p_matrix
        self._Z = z_matrix

        print("New starting position log-likelihood of: {:.3f}".format(log_likelihood))
        print("Finished exhaustive subsequence, starting EM")
        likelihood = self.run_em()

    def exhaustive_subsequence(self,iterations=1):
        """Initialize the EM algzorithm by using the exhaustive-subsequence method: run EM for a specified number of iterations on each possible length W subsequence of the input.

        Parameters
        ----------
        iterations : int
            Default: 1
            Number of iterations to run the EM algorithm on each distinct subsequence

        Returns
        -------
        p : array-like
            Matrix of probabities of letter c in column k of the highest likelihood starting position
            Additionally, sets self._p with this matrix

        """
        # Find total number of characters in Dataset:
        counts = collections.Counter(''.join(self._sequences))

        # Find unique set of length W subsequences so we don't have any redundencies
        unique = set()
        for sequence in self._sequences:
            for start in range(self._starting_pos):
                # Iterate through the starting positions and make a substring at the starting position and ending at that position plus the width
                unique.add(sequence[start:(start+self._width)])

        # Now that we have the set, we can iterate through all the unique sequences and set their p-matrix
        # Keep track of best log-likelihood
        best = float("-inf")
        best_seq = ''
        print("Total number of subsequences to evaluate: ",len(unique))
        evaluated = 0
#        my_j = 0
        for seq in unique:
#            if my_j < 500:
#                print(my_j)
#                my_j += 1
#                continue
            p_matrix = pd.DataFrame(data=np.zeros_like(self._P),index=['A','C','G','T'],columns=list(range(self._width + 1)))
            ############ Populate p_c_k, k > 0
            # Go through bases, and set probability of the found character at that position equal to 2/5, and the others to 1/5
            for i,base in enumerate(seq):
                for letter in p_matrix.index:
                    if letter == base:
                        p_matrix.loc[base,i+1] = (self._pseudo + 1)/(self._pseudo*4 + 1)
                    else:

                        p_matrix.loc[letter,i+1] = (self._pseudo)/(self._pseudo*4 + 1)

            ############ Populate p_c_k, k = 0
            n_0 = {}
            # Number of A's C's G's and T's in the sequence.
            counts_seq = collections.Counter(seq)
            for base in ['A','C','G','T']:
                # n_c_0 = total # of c's in dataset - count of c's in motif
                n_0[base] = (counts[base] - counts_seq[base])
            my_sum = sum(n_0.values())
            for i,base in enumerate(['A','C','G','T']):
                # p_matrix_0 = (n_c_0 + pseudocount)/sum n_b_0 + 4*pseudocount
                p_matrix.iloc[i,0] = (n_0[base] + self._pseudo)/(my_sum + self._pseudo*4)

            #TODO: Find what to do with background for now set all to .25
            # p_matrix.loc[:,0] = .25
            log_likelihood, p_matrix, z_matrix = self._run_em(p_matrix)
            evaluated += 1
            if log_likelihood > best:
                self._P = p_matrix
                self._Z = z_matrix
                best = log_likelihood
                best_seq = seq
            print("{:s} log-likelihood of: {:.3f}, best log-likelihood of: {:.3f}, {:s}.  {} left to evaluate".format(seq,log_likelihood,best,best_seq,len(unique) - evaluated))
        print("Finished exhaustive subsequence, starting EM")
        likelihood = self.run_em()

    # easy test passed
    def _calc_seq_prob(self,seq,z,p_matrix):
        """_calculate_seq_prob calculates the log probability of a sequence given a sequence (seq), the motif starting position (z), and a probability matrix (p_matrix)

        Parameters
        ----------
        seq : string
            Sequence of whcih to calculate the probability
        z : int
            Starting position of motif in sequence
        p_matrix : array-like
            probability matrix of character c in row k of matrix, where the 0th row is the background

        Raise
        -------
        ValueError when z > L - W + 1

        Returns
        -------
        log-probability of the sequence

        """
        if z > (len(seq) - (p_matrix.shape[1] - 1)):
            raise ValueError("z must be less than L - W")

        total = 0
        for i,base in enumerate(seq):
            # Background
            if i < z or i > (z+(p_matrix.shape[1]-1)-1):
                total += np.log(p_matrix.loc[base,0])
            else:
                # 0th index of the motif is at column 1 in the p_matrix so use i - z + 1
                total += np.log(p_matrix.loc[base,(i - z + 1)])
        return total

    # easy test passed
    def _calc_seq_prob_sum(self,seq,p_matrix):
        """Calculate the sum of sequence probabilities over all z \in {0,...,(L - W)}

        Parameters
        ----------
        seq : string
            Sequence to calculate sum on
        p_matrix : array-like
            probability matrix for character c in column k of the motif, with background being in column 0

        Returns
        -------
        tuple: (log-probability, [individual probabilities])
            [0]: log-probability of the sum of sequence probabilities for all z \in {0,...,(L - W)}
            [1]: list of the probability of the sequence at the individual starts

        """
        # Initialize total at -inf corresponding to 0 probability
        total = float("-inf")
        individual = [0]*(len(seq) - (p_matrix.shape[1] - 1) + 1)
        # Iterate through all starting positions: {0,...,(L - W)}
        for z in range(len(seq) - (p_matrix.shape[1] - 1) + 1):
            # log(x + y) = x' + log(1 + exp(y' - x')), where ' denotes log-space
            # x' is supposed to be larger, i.e. closer to 0, and it starts at 0 so it should be larger
            # Define the total as x and y as the new sequence to add on
            prob = self._calc_seq_prob(seq,z,p_matrix)
            individual[z] = prob
            #if z == 0:
            total = prob + np.log(1 + np.exp(total - prob))
            #else:
            #    total = total + np.log(1 + np.exp(prob - total))
        return (total,individual)

    def _calc_log_likelihood(self,p_matrix,z_matrix):
        """TODO: Docstring for _calc_log_likelihood.

        Parameters
        ----------
        p_matrix : TODO
        z_matrix : TODO

        Returns
        -------
        Log-likelihood of data and starting positions given current parameters

        """
        # Normalize z matrix so we use the max percentage as the starting position.
        # i.e. convert .1 .1 .1 .7 to 0 0 0 1
    #    for i,row in enumerate(z_matrix):
    #        max_index = np.argmax(row)
    #        z_matrix[i,:] = np.zeros_like(row)
    #        z_matriz[i,max_index] = 1

#        total = 0
#        n = len(self._sequences)
        m = z_matrix.shape[1]
#        # Enumerate all sequences
#        for i, seq in enumerate(self._sequences):
#            # Enumerate all starting positions
#            for j, col in enumerate(z_matrix[i,:]):
#                # Calculate sequence prob
#                total += (z_matrix[i,j]*self._calc_seq_prob(seq,j,p_matrix) + n*np.log(1/m))
        total = 0
        for i,seq in enumerate(self._sequences):
            total += self._calc_seq_prob_sum(seq,p_matrix)[0] - np.log(m)

        return total

    def run_em(self):
        """Run the EM algorithm until convergence

        Returns
        -------
        float:
            current log-likelihood

        """
        current = float("-inf")
        i = 1
        print(self._P)
        while(True):
            log_likelihood,p_matrix,z_matrix = self._run_em(self._P)
            i += 1
            if (current - log_likelihood > -.05) and (current -  log_likelihood < 0):
                current = log_likelihood
                print("Difference less than -.05.  EM converged.  Log-likelihood: ", log_likelihood)
                break
            elif log_likelihood > current:
                current = log_likelihood
                self._P = p_matrix
                self._Z = z_matrix
                print("Finished iteration: ", i," \tLog-likelihood: ",current)
            else:
                print(self._P)
                raise ValueError("log-likelihood not increasing, something is wrong...")
        return current

    def _run_em(self,p_matrix):
        """TODO: Docstring for _run_em.

            Parameters
            ----------
            p_matrix : array-like
                Current matrix of probabilities for character c at position k of the motif, index starting at 1 since index 0 is background
                        0    1    2 .... W
                    A  0.0  0.0
                    C  0.0  0.0
                    G  0.0  0.0
                    T  0.0  0.0

            Returns
            -------
            tuple of (log_likelihood, p_matrix, z_matrix)

        """
        ################### Initialization #############
        # Use local versions of p_matrix and z_matrix n_matrix
        _p = pd.DataFrame().reindex_like(self._P).fillna(0)
        _z = np.zeros_like(self._Z)
        _n = pd.DataFrame().reindex_like(self._P).fillna(0)

        ################### E-Step #####################
        # Compute Z_{i,j}^t
        seq_sums = [0]*len(self._sequences)
        for i,sequence in enumerate(self._sequences):
            sequence_sum,individuals = self._calc_seq_prob_sum(sequence,p_matrix)
            seq_sums[i] = sequence_sum
            # Set the values for the sequence by dividing the individual probability by the sum, using subtraction in log-space and normalize
            _z[i,:] = self._normalize([(x - sequence_sum) for x in individuals])

        ################### M-Step #####################

            ##### populate n_matrix
        for i, sequence in enumerate(self._sequences):
            for j, base in enumerate(sequence):
                ######### The following only calculates n_c,k where k > 0
                # Add to the sum across sequences of the sum over positions where c appears
                # For each position we want k to correspond to the number of positions that each character could be at if it were part of a motif.  i.e. the character at position 0 in the sequence can't be the second position in the motif.
                if j < (self._width - 1):
                    k = j + 1
                    my_range = range(1,k+1)
                elif j > (len(sequence) - self._width):
                    #k = width - (distance from the end)
                    k = self._width - ((len(sequence) - 1) - j)
                    my_range = range(k,self._width+1)
                else:
                    my_range = range(1,self._width+1)
                # my_range represents a range over all the positions k in a motif that the character could be in

                for k in my_range:
                    #j needs to equal where the starting position would be
                    # if k = 1, starting pos = j
                    # if k = 2, starting pos = j - 1
                    #starting_pos = j - (k - 1)
                    starting_pos = j - (k - 1)
        #            _n.loc[base,k] += _z[j,starting_pos]
                    _n.loc[base,k] += _z[i,starting_pos]

                ###### k = 0
                # Populate n_c,0 with n_C (total # of c's in data set)
                _n.loc[base,0] += 1
                # After this nested loop, have one loop to subtract the sume of n_c,k for all k
        for base in range(4):
            _n.iloc[base,0] -= np.sum(_n.iloc[base,1:])


            ##### calculate p_matrix from n_matrix
        for base in ['A','C','G','T']:
            for position in range(self._P.shape[1]):
                _p.loc[base,position] = (_n.loc[base,position] + self._pseudo)/(np.sum(_n.loc[:,position]) + self._pseudo*4)

        log_likelihood = self._calc_log_likelihood(_p,_z)

        return (log_likelihood, _p, _z)


################ Testing
#sequences = []
#with open('example1.txt', 'r') as infile:
#    for line in infile:
#        sequences.append(line.replace('\n',''))
#test = MEME(sequences, 3, 1)

## Main
# This is the main function provided for you.
# Define additional functions to implement MEME
def main(args):
    # Parse input arguments
    # It is generally good practice to validate the input arguments, e.g.,
    # verify that the input and output filenames are provided
    if args.sub_name == 'meme':
        seq_file_path = args.sequences_filename
        W = args.width
        model_file_path = args.model
        position_file_path = args.positions
        subseq_file_path = args.subseqs
        sequences = []
        with open(seq_file_path, 'r') as infile:
            for line in infile:
                sequences.append(line.replace('\n',''))
        my_meme = MEME(sequences, W, 1)
        if args.start and len(args.start) == args.width:
            if len(args.start) != args.width:
                raise ValueError('Start must be same length as width')
            my_meme.test(args.start)
        else:
            my_meme.exhaustive_subsequence()

        positions = []
        subseqs = []

        for i,seq in enumerate(my_meme.z):
            positions.append(np.argmax(seq))
            subseqs.append(sequences[i][np.argmax(seq):(np.argmax(seq) + W + 1)])

        with open(position_file_path,'w') as out_file:
            for position in positions:
                print(position, file=out_file)

        with open(subseq_file_path,'w') as out_file:
            for sequence in subseqs:
                print(sequence, file=out_file)

        with open(model_file_path, 'w') as out_file:
            print(my_meme.p.to_csv(header=False,sep='\t'),file=out_file)
            print(my_meme.p.to_csv(header=False,sep='\t',float_format="%.3f"))
        if args.graph:
            heights = seq_logo.seq_logo(my_meme.p)
            plots = [0]*my_meme.p.shape[0]
            for letter in range(my_meme.p.shape[0]):
                plots[letter] = plt.bar(list(range(my_meme.p.shape[1])),
                        my_meme.p.iloc[letter,:],bottom=(np.sum(my_meme.p.iloc[0:letter,:]) if letter != 0 else 0))[0]
            plt.legend(plots,('A','C','G','T'))
            plt.ylim([0,2])
            plt.ylabel('Bits')
            plt.show()



    elif args.sub_name == 'logo':
        model_filename = args.model_filename
        # Read file as csv with tab separator, treat the first column as the row names, and drop columns that are NaN
        model = pd.read_csv(model_filename,header=None,sep='\t',index_col=0).dropna(axis=1)
        heights = seq_logo.seq_logo(model)
        plots = [0]*model.shape[0]
        if args.graph:
            for letter in range(model.shape[0]):
                plots[letter] = plt.bar(list(range(model.shape[1])),
                        model.iloc[letter,:],bottom=(np.sum(model.iloc[0:letter,:]) if letter != 0 else 0))[0]
            plt.legend(plots,('A','C','G','T'))
            plt.ylim([0,2])
            plt.ylabel('Bits')
            plt.show()

        else:
            print(heights)


# Note: this syntax checks if the Python file is being run as the main program
# and will not execute if the module is imported into a different module
if __name__ == "__main__":
    # Note: this example shows named command line arguments.  See the argparse
    # documentation for positional arguments and other examples.
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(title='subcommands',dest='sub_name')
    parser_em = subparsers.add_parser('meme',description=__doc__,help='Run meme algorithm')
    parser_logo = subparsers.add_parser('logo',description=__doc__,help='Compute sequence logo')
    parser_logo.add_argument('model_filename',
                            help='model file path.',
                            type=str)
    parser_logo.add_argument('--graph',
                            help='Graph model?',
                            action='store_true',
                            default=False)
    parser_em.add_argument('sequences_filename',
                        help='sequences file path.',
                        type=str)
    parser_em.add_argument('--width',
                        help='width of the motif.',
                        type=int,
                        default=6)
    parser_em.add_argument('--graph',
                            help='Graph model?',
                            action='store_true',
                            default=False)
    parser_em.add_argument('--start',
                        help='Motif possibility to start with, must be same length as --width.',
                        type=str,
                        default=None)
    parser_em.add_argument('--model',
                        help='model output file path.',
                        type=str,
                        default='model.txt')
    parser_em.add_argument('--positions',
                        help='position output file path.',
                        type=str,
                        default='positions.txt')
    parser_em.add_argument('--subseqs',
                        help='subsequence output file path.',
                        type=str,
                        default='subseqs.txt')

    args = parser.parse_args()
    # Note: this simply calls the main function above, which we could have
    # given any name
    main(args)
