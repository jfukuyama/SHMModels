import csv
import StringIO
import numpy as np
import fnmatch

class ContextModel:
    """Contains the information about mutation probabilities from a context model."""
    def __init__(self, context_length, pos_mutating, csv_string):
        """
        context_length -- The motif sizes.
        pos_mutating -- The position of the target base within the motif.
        csv_string -- A csv with two columns, one containing the nucleotide
        context and the other containing the theta values.
        """
        self.context_length = context_length
        self.pos_mutating = pos_mutating
        self.pad_left = pos_mutating
        self.pad_right = context_length - pos_mutating - 1
        if csv_string is not None:
            self.create_context_dict(csv_string)

    def create_context_dict(self, csv_string):
        """makes a dictionary, keys are contexts and values are probability of mutation in one unit of time"""
        reader = csv.reader(StringIO.StringIO(csv_string), delimiter=',')
        # skip the header
        next(reader, None)
        d = dict(reader)
        # change from theta_0 + theta_i to logit(theta_0 + theta_i)
        d_out = {}
        for k in d.keys():
            d_out[k.upper()] = _logistic(float(d[k]))
        self.context_dict = d_out

    def get_context_prob(self, idx, sequence):
        if(sequence[idx] != "C"):
            return 0
        # for positions on the flanks, we compute the marginal
        # probabilities as needed
        context = self.get_context(idx, sequence)
        if context in self.context_dict.keys():
            return self.context_dict[context]
        elif self.in_flank(idx, len(sequence)):
            # compute the marginal probability, add to dict, and return
            self.context_dict[str(context)] = self.compute_marginal_prob(context)
            return self.context_dict[context]
        else:
            raise ValueError("The context model doesn't have an entry for the context " + context)

    def get_context(self, idx, sequence):
        if self.in_flank(idx, len(sequence)):
            sequence = ["N"] * self.pad_left + sequence + ["N"] * self.pad_right
            idx = idx + self.pad_left
        return "".join(sequence[(idx-self.pad_left):(idx+self.pad_right+1)])

    def in_flank(self, idx, seq_len):
        if idx < self.pos_mutating:
            return True
        if idx > (seq_len - (self.context_length - self.pos_mutating)):
            return True
        return False

    def compute_marginal_prob(self, context):
        """Computes marginal probabilities"""
        search_string = []
        # replace all the n's with ? to create the search string
        for c in list(context):
            if c == "N":
                search_string.append("?")
            else:
                search_string.append(c)
        search_string = "".join(search_string)
        # find all the keys with contexts consistent with the n-ful context
        consistent_contexts = fnmatch.filter(self.context_dict.keys(), search_string)
        # remove any contexts that already have n's (already
        # marginalized contexts)
        consistent_contexts = [cc for cc in consistent_contexts
                               if not "N" in cc]
        # get the probabilities for all the consistent contexts
        cc_probs = [self.context_dict[k] for k in consistent_contexts]
        return np.mean(cc_probs)


def _logistic(x):
    """The logistic function"""
    return 1. / (1 + np.exp(-x))
