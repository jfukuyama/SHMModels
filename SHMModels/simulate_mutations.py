import numpy as np
import numpy.random
import GPy
import pkgutil
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
from SHMModels.summary_statistics import write_all_stats
from SHMModels.fitted_models import ContextModel
#from timeit import default_timer as timer


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
    n_time_bins -- The number of bins for discretizing mutation times.

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
                 n_time_bins = 100):
        """Returns a MutationProcess object with a specified start_seq"""
        if not isinstance(start_seq, Seq):
            raise TypeError("The input sequence must be a Seq object")
        # we're going to need reverse complements, so the alphabet is important
        if not isinstance(start_seq.alphabet, type(IUPAC.unambiguous_dna)):
            raise TypeError("The alphabet must be IUPAC.unambiguous_dna")
        self.start_seq = start_seq
        self.ber_lambda = ber_lambda
        self.mmr_lambda = mmr_lambda
        self.exo_params = exo_params
        self.pol_eta_params = pol_eta_params
        self.ber_params = ber_params
        self.aid_context_model = aid_context_model
        self.gp_lengthscale = gp_lengthscale
        self.NUCLEOTIDES = ["A", "G", "C", "T"]
        self.n_time_bins = n_time_bins

    def generate_mutations(self):
        self.sample_lesions()
        #self.sample_repair_types()
        #self.sample_repaired_sequence()

    def sample_lesions(self):
        """Sample lesions induced by AID"""
        self.aid_lesions = make_aid_lesions(self.start_seq,
                                            context_model=self.aid_context_model,
                                            gp_lengthscale=self.gp_lengthscale,
                                            n_time_bins=self.n_time_bins)

    def sample_repair_types(self):
        """Sample a repaired sequence given a base sequence and lesions."""
        # first get waiting times to recruit the BER/MMR machinery
        ber_wait_times = self.sample_ber_wait_times()
        mmr_wait_times = self.sample_mmr_wait_times()
        exo_positions = self.sample_exo_positions()
        self.repair_types = create_repair_types(self.aid_lesions,
                                                ber_wait_times,
                                                mmr_wait_times,
                                                exo_positions)

    def sample_repaired_sequence(self):
        int_seq = self.start_seq
        # choose either the fw or rc strand to sample
        strand = np.random.choice([0, 1], size=1, p=[self.p_fw, 1-self.p_fw])[0]
        repairs = list(self.repair_types[strand])
        while len(repairs) > 0:
            # get the next lesion to repair and how to repair it,
            # update the list of resolutions to remove it and any
            # others that are resolved in the process.
            (rt, repairs) = get_next_repair(repairs)
            # add the info about exo length here
            if(rt.repair_type == "mmr"):
                self.mmr_sizes = self.mmr_sizes + [rt.exo_hi - rt.exo_lo]
            if strand == 0:
                int_seq = self.sample_sequence_given_repair(int_seq, rt)
            elif strand == 1:
                int_seq = self.sample_sequence_given_repair(int_seq.reverse_complement(), rt).reverse_complement()
            else:
                raise ValueError("Something went wrong, strand should be either 0 or 1, it was " + strand)
        self.repaired_sequence = int_seq

    def sample_sequence_given_repair(self, sequence, r):
        """Samples an intermediate sequence given input and repair type.

        Keyword arguments:
        sequence -- A Seq object containing the input sequence.
        r -- A Repair object describing the repair.

        Returns: A new sequence.
        """
        # so we can replace elements of the string
        s = list(str(sequence))
        if r.repair_type == "replicate":
            s[r.idx] = "T"
        elif r.repair_type == "ber":
            s[r.idx] = self.sample_ber()
        elif r.repair_type == "mmr":
            s = self.sample_pol_eta(s, r.exo_lo, r.exo_hi)
            
        s = "".join(s)
        return Seq(s, alphabet=IUPAC.unambiguous_dna)

    def sample_ber(self):
        """Samples a nucleotide as repaired by BER.

        Returns: A nucleotide.
        """
        return np.random.choice(self.NUCLEOTIDES, size=1, p=self.ber_params)[0]

    def sample_pol_eta(self, seq, lo, hi):
        """Samples sequences repaired by pol eta

        Keyword arguments:
        seq -- A list of characters describing the base sequence.
        lo -- The index of the most 5' nucleotide to be sampled.
        hi -- The index of the most 3' nucleotide to be sampled.

        Returns: A list of characters describing the sampled sequence.
        """
        new_seq = seq
        for i in range(lo, hi + 1):
            new_seq[i] = np.random.choice(self.NUCLEOTIDES, size=1, p=self.pol_eta_params[seq[i]])[0]
        return new_seq

    def sample_ber_wait_times(self):
        # for every lesion, sample a random exponential with rate
        # parameter ber_lambda
        return((np.random.exponential([1. / self.ber_lambda for
                                       _ in self.aid_lesions[0]]),
                np.random.exponential([1. / self.ber_lambda for
                                       _ in self.aid_lesions[1]])))

    def sample_mmr_wait_times(self):
        # for every lesion, sample a random exponential with rate parameter mmr_lambda
        return((np.random.exponential([1. / self.mmr_lambda for
                                       _ in self.aid_lesions[0]]),
                np.random.exponential([1. / self.mmr_lambda for
                                       _ in self.aid_lesions[1]])))

    def sample_exo_positions(self):
        l = len(self.start_seq)
        exo_positions = ([(max(0, a - np.random.geometric(self.exo_params['left'])),
                           min(a + np.random.geometric(self.exo_params['right']), l - 1)) for
                          a in self.aid_lesions[0]],
                         [(max(0, a - np.random.geometric(self.exo_params['left'])),
                           min(a + np.random.geometric(self.exo_params['right']), l - 1)) for
                          a in self.aid_lesions[1]])
        return(exo_positions)

class Repair(object):
    """Describes how a lesion is repaired. A Repair object has the following properties:

    Attributes:
    idx -- The location of the lesion.
    repair_type -- The type of repair machinery recruited first.
    repair_time -- The time at which the repair machinery is recruited.
    exo_lo -- The position of the most 3' base stripped out by EXO1.
    exo_hi -- The position of the most 5' base stripped out by EXO1.

    """
    def __init__(self, idx, repair_type, repair_time, exo_lo, exo_hi):
        self.idx = idx
        self.time = repair_time
        self.repair_type = repair_type
        if repair_type == "mmr":
            self.exo_lo = exo_lo
            self.exo_hi = exo_hi


def create_repair_types(aid_lesions, ber_wait_times, mmr_wait_times, exo_positions, replication_time):
    """Creates repair types from recruitment times for repair machinery.

    Keyword arguments:
    aid_lesions -- Two lists, first containing the indices of aid
    lesions on the fw strand, second containing the indices of aid
    lesions on the rc strand.
    ber_wait_times -- Two lists, giving recruitment times of ber
    machinery to each of the aid lesions.
    mmr_wait_times -- Two lists, giving recruitment times of the mmr
    machinery to each of the aid lesions.
    exo_positions -- Two lists, giving the indices of the 5'-most and
    3'-most bases that would be stripped out if each lesion were
    repaired by mmr.
    replication_time -- If no repair machinery is recruited by this
    time, the lesion gets replicated over.

    Returns: Two lists of Repair objects describing the first type of
    repair machinery recruited to each lesion and how it will act.

    """
    repairs = ([], [])
    for strand in [0, 1]:
        zipped = zip(aid_lesions[strand],
                     ber_wait_times[strand],
                     mmr_wait_times[strand],
                     exo_positions[strand])
        for (idx, bwt, mwt, el) in zipped:
            if replication_time < bwt and replication_time < mwt:
                repairs[strand].append(Repair(idx=idx,
                                              repair_type="replicate",
                                              repair_time=replication_time,
                                              exo_lo=None,
                                              exo_hi=None))
            elif bwt < mwt:
                repairs[strand].append(Repair(idx=idx,
                                              repair_type="ber",
                                              repair_time=bwt,
                                              exo_lo=None,
                                              exo_hi=None))
            else:
                repairs[strand].append(Repair(idx=idx,
                                              repair_type="mmr",
                                              repair_time=mwt,
                                              exo_lo=el[0],
                                              exo_hi=el[1]))
    return repairs


def get_next_repair(repair_list):
    """Describes repair types for a set of lesions

    Keyword arguments:
    repair_list -- A list of Repair objects.

    Returns: A tuple giving the index and repair type of the next
    lesion to repair along with the remaining lesions.

    """
    (next_repair_time, next_repair, next_repair_idx) = \
        min([(val.time, val, idx) for (idx, val) in enumerate(repair_list)])
    new_repair_list = list(repair_list)
    if next_repair.repair_type == "mmr":
        # we only keep repairs that are outside of the range of exo
        new_repair_list = [r for r in new_repair_list if
                           r.idx < next_repair.exo_lo or
                           r.idx > next_repair.exo_hi]
        return (next_repair, new_repair_list)
    else:
        new_repair_list.pop(next_repair_idx)
        return(next_repair, new_repair_list)


def make_aid_lesions(sequence, context_model, gp_lengthscale, n_time_bins):
    """Simulates AID lesions on a sequence

    Keyword arguments:
    sequence -- A Seq object using the IUPAC Alphabet
    context_model -- A model giving relative rates of deamination in
    different nucleotide contexts.
    gp_lengthscale -- A list with one element for the space
    lengthscale and one for the time lengthscale.
    n_time_bins -- How many bins to discretize the unit interval into.

    Returns: A pair of vectors, the first giving the indices of lesions
    on the forward strand and the second giving the indices of lesions
    on the reverse complement.

    """
    if not isinstance(sequence, Seq):
        raise TypeError("The input sequence must be a Seq object")
    if not isinstance(sequence.alphabet, type(IUPAC.unambiguous_dna)):
        raise TypeError("The input sequence must have an IUPAC.unambiguous_dna alphabet")
    # Create a matrix describing a draw from the Gaussian process prior
    gp_array = draw_from_gp(len(sequence), n_time_bins, gp_lengthscale)
    base_rate_array = make_base_rate_array(sequence, context_model, n_time_bins)
    rate_array = np.exp(gp_array) * base_rate_array
    lesions = np.random.poisson(rate_array)
    return lesions

def draw_from_gp(seq_length, n_time_bins, gp_lengthscale):
    """Creates a matrix describing a draw from a GP

    Keyword arguments:
    seq_length --- The length of the sequence.
    n_time_bins --- The number of time bins for discretization.
    gp_lengthscale --- A dictionary with one element for the space
    lengthscale and another element for the time lengthscale.

    Returns:
    An array of size 2 x seq_length x n_time_bins

    """
    # The kernel overall is a product of one over the sequence and one over time
    k_seq = GPy.kern.RBF(input_dim = 1, lengthscale = gp_lengthscale["space"], active_dims = [0])
    k_time = GPy.kern.RBF(input_dim = 1, lengthscale = gp_lengthscale["time"], active_dims = [1])
    k = k_seq * k_time
    # Draw from the GP
    seqs, times = np.mgrid[0:seq_length, 0:n_time_bins]
    times = times / float(n_time_bins)
    X = np.vstack((seqs.flatten(), times.flatten())).T
    K = k.K(X)
    s = np.random.multivariate_normal(np.zeros(X.shape[0]), K)
    seq_and_time_draw = s.reshape(*seqs.shape)
    # Output is repeated, one for fw strand and one for rc
    return np.array([seq_and_time_draw, seq_and_time_draw])


def make_base_rate_array(sequence, context_model, n_time_bins):
    """Creates a matrix giving the AID deamination rates at every position
in the sequence

    Keyword arguments:
    sequence --- The sequence that will accumulate lesions.
    context_model --- A context model giving probabilities of AID lesions by context.
    n_time_bins --- Number of time bins for discretization.

    Returns:
    An array of size 2 x length(sequence) x n_time_bins
    """
    fw_probs = [context_model.get_context_prob(i, sequence) for i in range(len(sequence))]
    rc_probs = [context_model.get_context_prob(len(sequence) - i - 1, sequence.reverse_complement()) for i in range(len(sequence))]
    print fw_probs
    print rc_probs
    fw_rates = [-np.log(1 - p) for p in fw_probs]
    rc_rates = [-np.log(1 - p) for p in rc_probs]
    sequence_rates = np.array([fw_rates, rc_rates])
    sequence_rates.shape = [sequence_rates.shape[0], sequence_rates.shape[1], 1]
    sequence_and_time_rates = np.tile(sequence_rates, [1,1,n_time_bins])
    return sequence_and_time_rates

def simulate_sequences_abc(germline_sequence,
                           aid_context_model,
                           context_model_length,
                           context_model_pos_mutating,
                           n_seqs,
                           n_mutation_rounds,
                           ss_file,
                           param_file,
                           sequence_file,
                           n_sims,
                           write_ss=True,
                           write_sequences=False):

    sequence = list(SeqIO.parse(germline_sequence, "fasta",
                                alphabet=IUPAC.unambiguous_dna))[0]
    aid_model_string = pkgutil.get_data("SHMModels", aid_context_model)
    aid_model = ContextModel(context_model_length,
                             context_model_pos_mutating,
                             aid_model_string)
    n_sum_stats = 310
    n_params = 9
    ss_array = np.zeros([n_sims, n_sum_stats])
    param_array = np.zeros([n_sims, n_params])
    mutated_seq_array = np.empty([n_sims * n_seqs, 2], dtype="S500")
    for sim in range(n_sims):
        mutated_seq_list = []
        mmr_length_list = []
        # the prior specification
        #start_prior = timer()
        ber_lambda = np.random.uniform(0, 1, 1)[0]
        bubble_size = np.random.randint(5, 50)
        exo_left = 1 / np.random.uniform(1, 50, 1)[0]
        exo_right = 1 / np.random.uniform(1, 50, 1)[0]
        pol_eta_params = {
                     'A': [.9, .02, .02, .06],
                     'G': [.01, .97, .01, .01],
                     'C': [.01, .01, .97, .01],
                     'T': [.06, .02, .02, .9]
        }
        ber_params = np.random.dirichlet([1, 1, 1, 1])
        p_fw = np.random.uniform(0, 1, 1)[0]
        #end_prior = timer()
        #start_seqs = timer()
        for i in range(n_seqs):
            mr = MutationRound(sequence.seq,
                               ber_lambda=ber_lambda,
                               mmr_lambda=1 - ber_lambda,
                               replication_time=100,
                               bubble_size=bubble_size,
                               aid_time=10,
                               exo_params={'left': exo_left,
                                           'right': exo_right},
                               pol_eta_params=pol_eta_params,
                               ber_params=ber_params,
                               p_fw=p_fw,
                               aid_context_model=aid_model)
            for j in range(n_mutation_rounds):
                mr.mutation_round()
                mr.start_seq = mr.repaired_sequence
            mutated_seq_list.append(SeqRecord(mr.repaired_sequence, id=""))
            if(len(mr.mmr_sizes) > 0):
                mmr_length_list.append(np.mean(mr.mmr_sizes))
        #end_seqs = timer()
        #start_ss = timer()
        if write_ss:
            ss_array[sim, :] = write_all_stats(sequence,
                                            mutated_seq_list,
                                            np.mean(mmr_length_list),
                                            file=None)
        params = [ber_lambda, bubble_size, exo_left, exo_right, ber_params[0], ber_params[1], ber_params[2], ber_params[3], p_fw]
        param_array[sim, :] = params
        if write_sequences:
            seq_strings = [str(ms.seq) for ms in mutated_seq_list]
            mutated_seq_array[(sim * n_seqs):((sim + 1) * n_seqs),0] = seq_strings
            mutated_seq_array[(sim * n_seqs):((sim + 1) * n_seqs),1] = sim
        #end_ss = timer()
        #print("Draw the prior: {}, Simulate sequences: {}, Write summary statistics: {}".format(end_prior - start_prior, end_seqs - start_seqs, end_ss - start_ss))
    np.savetxt(param_file, param_array, delimiter=",")
    if write_ss:
        np.savetxt(ss_file, ss_array, delimiter=",")
    if write_sequences:
        np.savetxt(sequence_file, mutated_seq_array, delimiter=",", fmt="%s")
    return (param_array, ss_array)
