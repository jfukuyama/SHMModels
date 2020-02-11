import numpy as np
import numpy.random
import GPy
import pkgutil
import copy
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from SHMModels.summary_statistics import write_all_stats
from SHMModels.fitted_models import ContextModel

class MutationProcess(object):
    """Simulates the mutation process. A  MutationProcess has the following properties:

    Attributes:
    start_seq -- The initial sequence.
    aid_lesions -- The locations and times of AID lesions.
    repair_types -- The locations, types, and times of the repairs.
    pol_eta_params -- A dictionary, keyed by nucleotide, each element
    describing the probability of Pol eta creating a mutation from
    that nucleotide to another nucleotide.
    ber_params -- A vector describing the probability that the BER
    machinery incorporates each nucleotide.
    exo_params -- Parameters determining the number of bases exo1
    strips out to the left and to the right.
    """

    def __init__(self, start_seq,
                 ber_lambda=1,
                 mmr_lambda=1,
                 exo_params={'left': .2, 'right': .2},
                 pol_eta_params={
                     'A': [1, 0, 0, 0],
                     'G': [0, 1, 0, 0],
                     'C': [0, 0, 1, 0],
                     'T': [0, 0, 0, 1]
                 },
                 ber_params=[0, 0, 1, 0],
                 aid_context_model=None,
                 gp_lengthscale={'space': 10, 'time': .2},
                 overall_rate = 1):
        """Returns a MutationProcess object with a specified start_seq"""
        if not isinstance(start_seq, Seq):
            raise TypeError("The input sequence must be a Seq object")
        # we're going to need reverse complements, so the alphabet is important
        if not isinstance(start_seq.alphabet, type(IUPAC.unambiguous_dna)):
            raise TypeError("The alphabet must be IUPAC.unambiguous_dna")
        # store the sequence as a 2 x len(sequence) array for forward and complement
        self.start_seq = np.array([list(str(start_seq)), list(str(start_seq.complement()))])
        self.seq_len = self.start_seq.shape[1]
        self.ber_lambda = ber_lambda
        self.mmr_lambda = mmr_lambda
        self.exo_params = exo_params
        self.pol_eta_params = pol_eta_params
        self.ber_params = ber_params
        self.aid_context_model = aid_context_model
        self.gp_lengthscale = gp_lengthscale
        self.NUCLEOTIDES = ["A", "G", "C", "T"]
        self.overall_rate = overall_rate
        self.aid_lesions_per_site = np.zeros((2, self.seq_len))

    def generate_mutations(self):
        self.sample_lesions()
        self.sample_repairs()
        self.sample_repaired_sequence()

    def sample_lesions(self):
        """Sample lesions induced by AID"""
        self.aid_lesions = make_aid_lesions(self.start_seq,
                                            context_model=self.aid_context_model,
                                            gp_lengthscale=self.gp_lengthscale,
                                            overall_rate = self.overall_rate)

    def sample_repairs(self):
        """Sample repairs for every AID lesion."""
        # first get waiting times to recruit the BER/MMR machinery
        self.repair_types = []
        for row in range(self.aid_lesions.shape[0]):
            lesion = self.aid_lesions[row,:]
            self.repair_types.append(self.sample_one_repair(lesion[0], lesion[1], lesion[2]))

    def sample_one_repair(self, strand, location, aid_time):
        # repair type is ber w.p. lambda_b / (lambda_b + lambda_m)
        if np.random.uniform() <= (self.ber_lambda / (self.ber_lambda + self.mmr_lambda)):
            repair_type = "ber"
        else:
            repair_type = "mmr"
        # repair time is aid_time + exponential(lambda_b + lambda_m)
        repair_time = aid_time + np.random.exponential(self.ber_lambda + self.mmr_lambda)
        # exo_hi is a truncated version of a geometric distribution
        exo_left = np.random.geometric(self.exo_params['left'])
        exo_right = np.random.geometric(self.exo_params['right'])
        if strand == 0:
            exo_lo = max(0, location - exo_left)
            exo_hi = min(self.seq_len - 1, location + exo_right)
        else:
            exo_lo = max(0, location - exo_right)
            exo_hi = min(self.seq_len - 1, location + exo_left)
        return Repair(strand, location, aid_time, repair_type, repair_time, exo_lo, exo_hi)

    def sample_repaired_sequence(self):
        intermediate_seq = copy.copy(self.start_seq)
        ## sort the repairs in the order they occur
        self.repair_types.sort(key = lambda x: x.repair_time)
        for r in self.repair_types:
            # sample a new value for intermediate_seq and remove any
            # repairs that no longer should occur
            self.process_one_repair(r, intermediate_seq)
        self.repaired_sequence = intermediate_seq

    def process_one_repair(self, repair, sequence):
        """Samples an intermediate sequence from a repair and updates the list of repairs based on the intermediate sequence

        Keyword arguments:
        repair -- A repair object, describing the location and type of the repair
        sequence -- The sequence to be repaired
        """
        exo_strips = []
        c_mutations = []
        strand = repair.strand
        idx = repair.idx
        if repair.repair_type == "ber":
            self.aid_lesions_per_site[strand,idx] = self.aid_lesions_per_site[strand, idx] + 1
            sequence[strand][idx] = self.sample_ber()
            if sequence[strand][idx] != "C":
                c_mutations.append((strand, idx))
        elif repair.repair_type == "mmr":
            self.aid_lesions_per_site[strand,idx] = self.aid_lesions_per_site[strand, idx] + 1
            for i in range(repair.exo_lo, repair.exo_hi + 1):
                old_base = sequence[strand][i]
                new_base = self.sample_pol_eta(old_base)
                sequence[strand][i] = new_base
                if (old_base == "C") & (new_base != "C"):
                    c_mutations.append((strand, i))
                exo_strips.append((strand, i))
        ## Mark repairs that will no longer happen
        # exo_strips and c_mutations are lists, elements are tuples of
        # (strand, idx) corresponding to locations of exo stripping or
        # locations where a c mutated to something else
        self.update_repairs(exo_strips, c_mutations, repair.repair_time)

    def update_repairs(self, exo_strips, c_mutations, repair_time):
        # if the repair has an AID time later than the repair time
        # and the current repair changed a C at that position to
        # something else, set the repair to None
        for r in self.repair_types:
            if(r.repair_type != "none"):
                if (r.aid_time > repair_time) & ((r.strand, r.idx) in c_mutations):
                    r.repair_type = "none"
                ## If the repair has an AID time earlier than the repair
                ## time and the current repair's exo window covered the
                ## lesion position, set the repair to None
                elif (r.aid_time < repair_time) & ((r.strand, r.idx) in exo_strips):
                    r.repair_type = "none"

    def sample_ber(self):
        """Samples a nucleotide as repaired by BER.

        Returns: A nucleotide.
        """
        return np.random.choice(self.NUCLEOTIDES, size=1, p=self.ber_params)[0]

    def sample_pol_eta(self, old_base):
        """Samples a nucleotide as repaired by pol eta.

        Returns: A nucleotide.
        """
        return np.random.choice(self.NUCLEOTIDES, size = 1, p=self.pol_eta_params[old_base])[0]

class Repair(object):
    """Describes how a lesion is repaired. A Repair object has the following properties:

    Attributes:
    idx -- The location of the lesion.
    aid_time -- The time of the AID lesion.
    repair_type -- The type of repair machinery recruited first, or
    'none' if the repair was made impossible by a prior repair.
    repair_time -- The time at which the repair machinery is recruited.
    exo_lo -- The position of the left-most base stripped out by EXO1.
    exo_hi -- The position of the right-most base stripped out by
    EXO1. The bases stripped out are [strand][idx][exo_lo:exo_hi]

    """
    def __init__(self, strand, idx, aid_time, repair_type, repair_time, exo_lo, exo_hi):
        self.strand = int(strand)
        self.idx = int(idx)
        self.aid_time = aid_time
        self.repair_time = repair_time
        self.repair_type = repair_type
        if repair_type == "mmr":
            self.exo_lo = int(exo_lo)
            self.exo_hi = int(exo_hi)

def make_aid_lesions(sequence, context_model, gp_lengthscale, overall_rate):
    """Simulates AID lesions on a sequence

    Keyword arguments:
    sequence -- An array with characters for nucleotides, dimension 2
    x sequence length.
    context_model -- A model giving relative rates of deamination in
    different nucleotide contexts.
    gp_lengthscale -- A list with one element for the space
    lengthscale and one for the time lengthscale.
    overall_rate -- A scalar controlling the overall rate of the PP.

    Returns: A matrix with number of rows equal to number of
    lesions. First column gives the strand, second column gives the
    location along the sequence, and third column gives the time of
    the lesion.

    """
    # Create a matrix describing a draw from the Gaussian process prior
    base_rate_array = make_base_rate_array(sequence, context_model, overall_rate = overall_rate)
    pp_draw = draw_poisson_process_from_base_rate(base_rate_array)
    gp_draw = draw_from_gp(input_points = pp_draw, gp_lengthscale = gp_lengthscale)
    points_to_keep = [np.random.uniform(0.0, 1.0, 1)[0] <= sigmoid(g) for g in gp_draw]
    return pp_draw[points_to_keep,:]

def draw_from_gp(input_points, gp_lengthscale):
    """Creates a matrix describing a draw from a GP

    Keyword arguments:
    input_points --- A J x 3 matrix, giving strand, index, and time of
    the lesions.
    gp_lengthscale --- A dictionary with one element for the space
    lengthscale and another element for the time lengthscale.

    Returns:
    A vector of length J

    """
    # if input_points has length 0,
    if input_points.shape[0] == 0:
        return np.zeros(shape = (0,3))
    # The kernel overall is a product of one over the sequence and one over time
    k_seq = GPy.kern.RBF(input_dim = 1, lengthscale = gp_lengthscale["space"], active_dims = [0])
    k_time = GPy.kern.RBF(input_dim = 1, lengthscale = gp_lengthscale["time"], active_dims = [1])
    k = k_seq * k_time
    X = input_points[:,[1,2]]
    K = k.K(X)
    gp_draw = np.random.multivariate_normal(np.zeros(X.shape[0]), K)
    return gp_draw


def make_base_rate_array(sequence, context_model, overall_rate):
    """Creates a matrix giving the AID deamination rates at every position
in the sequence

    Keyword arguments:
    sequence --- The sequence that will accumulate lesions.
    context_model --- A context model giving probabilities of AID lesions by context.
    overall_rate -- A scaling factor to increase or decrease the PP rate.

    Returns:
    An array of size 2 x length(sequence)
    """
    seq_len = sequence.shape[1]
    # lesion probabilities for the fw strand
    fw_probs = [context_model.get_context_prob(i, list(sequence[0,:])) for i in range(seq_len)]
    # sequence[1,:] is the complementary strand, we need the reverse complement for the context model
    rc_sequence = list(reversed(sequence[1,:]))
    rc_probs = [context_model.get_context_prob(i, rc_sequence) for i in range(seq_len)]
    # convert to rates. We are storing the fw strand and the
    # complementary strand, not the reverse complementary strand, and
    # so we need to re-reverse the rc_probs list we made before.
    fw_rates = [-np.log(1 - p) for p in fw_probs]
    c_rates = list(reversed([-np.log(1 - p) for p in rc_probs]))
    # make the array with the rates over space, promote it so it has
    # an extra dimension for time, and fill in over n_time_bins with a
    # constant set of rates
    sequence_rates = np.array([fw_rates, c_rates]) * overall_rate
    return sequence_rates

def draw_poisson_process_from_base_rate(base_rate_array):
    """Draws from a Poisson Process

    Keyword arguments:
    base_rate_array -- A 2 x seq_len matrix, giving the rates for the
    Poisson process at each position in the sequence.

    Returns: A J x 3 matrix, rows corresponding to points sampled from
    the Poisson process, first column strand, second column location
    along the sequence, third column time.

    """
    ## Number of lesions per site comes from a Poisson with rate given by base_rate_array
    n_lesions_per_site = np.zeros(base_rate_array.shape)
    for idx, value in np.ndenumerate(base_rate_array):
        n_lesions_per_site[idx[0], idx[1]] = np.random.poisson(lam = value, size = 1)
    ## Each row of the lesion matrix is a 3-vector of [strand,
    ## location on sequence, time] corresponding to one lesion
    lesion_matrix = np.zeros((int(sum(sum(n_lesions_per_site))), 3))
    ## For each site, we sample times uniformly in [0,1] for each of
    ## the n_lesions and put each one in the lesion_matrix
    row = 0
    for idx, n_lesions in np.ndenumerate(n_lesions_per_site):
        lesion_times = np.random.uniform(0.0, 1.0, int(n_lesions))
        for t in lesion_times:
            lesion_matrix[row, :] = [idx[0], idx[1], t]
            row = row + 1
    return lesion_matrix

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
