import unittest
import numpy as np
import pkgutil
import copy
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from SHMModels.simulate_mutations import *
from SHMModels.fitted_models import ContextModel


class testRepairs(unittest.TestCase):

    def setUp(self):
        ## set up a mp with deterministic ber and pol eta parameters
        naive_seq = Seq("CGCA", alphabet=IUPAC.unambiguous_dna)
        cm = aid_context_model = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
        ## BER always changes C to A
        ber_params=[1, 0, 0, 0]
        ## here pol eta always changes C to T
        pol_eta_params={
            'A': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'C': [0, 0, 0, 1],
            'T': [0, 0, 0, 1]
        }

        self.mp = MutationProcess(
            naive_seq,
            pol_eta_params = pol_eta_params,
            ber_params = ber_params,
            aid_context_model = cm)
        self.mp.sample_lesions()
        self.mp.sample_repairs()

    def test_sample_repairs(self):
        ## check that we have the right number of repairs (sum of
        ## aid_lesions matrix is the same length as the repair types
        ## object)
        self.assertEqual(self.mp.aid_lesions.shape[0], len(self.mp.repair_types))

    def test_sample_one_repair(self):
        pass

    def test_sample_repaired_sequence_ber(self):
        ## in our test sequence, there is a C at the third position on
        ## the forward strand (so idx = 2 because we are zero indexing)
        strand = 0
        idx = 2
        aid_time = .5
        repair_time = .6
        exo_lo = 0
        exo_hi = 2
        ## set the MP to just have one repair, due to BER
        one_ber_repair = Repair(strand = strand, idx = idx, aid_time = aid_time,
                                repair_type = "ber",
                                repair_time = repair_time,
                                exo_lo = exo_lo, exo_hi = exo_hi)
        self.mp.repair_types = [one_ber_repair]
        self.mp.sample_repaired_sequence()
        ## we set up so that BER always changes C to A
        self.assertEqual(self.mp.repaired_sequence[strand,idx], "A")

    def test_sample_repaired_sequence_mmr(self):
        ## in our test sequence, there is a C at the third position on
        ## the forward strand (so idx = 2 because we are zero indexing)
        strand = 0
        idx = 2
        aid_time = .5
        repair_time = .6
        exo_lo = 0
        exo_hi = 2
        ## Set up a repair object for MMR with a certain exo window
        ## and check that the sampled sequence is correct.
        one_mmr_repair = Repair(strand = strand, idx = idx, aid_time = aid_time,
                                repair_type = "mmr",
                                repair_time = repair_time,
                                exo_lo = exo_lo, exo_hi = exo_hi)
        self.mp.repair_types = [one_mmr_repair]
        self.mp.sample_repaired_sequence()
        ## we set up so that pol eta always changes C to T
        ## the nucleotide at position idx is the C corresponding to the lesion
        self.assertEqual(self.mp.repaired_sequence[strand,idx], "T")
        ## the nucleotide at position 0 is another C that is in the exo window
        self.assertEqual(self.mp.repaired_sequence[strand,0], "T")

    def test_repair_processing_c_mutated(self):
        ## If at a certain position along the sequence, we draw a
        ## potential lesion at times t1 and t2 and if the C base was
        ## changed to another base at time t with t1 < t < t2, then
        ## the lesion at time t2 doesn't recruit any machinery and
        ## should be excluded.

        ## Set up an mp object with some repairs
        strand = 0; idx = 1
        aid_time_1 = 1; repair_time_1 = 2; aid_time_2 = 3; repair_time_2 = 4
        r1 = Repair(strand = strand, idx = idx,
                    aid_time = aid_time_1, repair_time = repair_time_1,
                    repair_type = "ber", exo_lo = np.nan, exo_hi = np.nan)
        r2 = Repair(strand = strand, idx = idx,
                    aid_time = aid_time_2, repair_time = repair_time_2,
                    repair_type = "ber", exo_lo = np.nan, exo_hi = np.nan)
        self.mp.repair_types = [r1, r2]
        self.mp.process_one_repair(r1, self.mp.start_seq)
        ## We set up so that BER deterministically mutates away from
        ## C, so the first lesion made the second impossible
        self.assertEqual(self.mp.repair_types[1].repair_type, "none")

    def test_repair_processing_lesion_stripped(self):
        ## If we have the following set of lesions and repairs:
        ##      A
        ##         A
        ##    --M------

        ## where we have two AID lesions, the first lesion repaired by
        ## MMR, and the MMR machinery strips out a region including
        ## the second lesion, the second lesion doesn't recruit any
        ## machinery and should be excluded.

        ## Set up an mp object with some repairs
        ## on the forward strand
        def repair_processing_results(aid_time_1, aid_time_2, repair_time_1, repair_time_2):
            mp = copy.deepcopy(self.mp)
            mp.pol_eta_params = {
                'A': [1, 0, 0, 0],
                'G': [0, 1, 0, 0],
                'C': [0, 0, 1, 0],
                'T': [0, 0, 0, 1]
            }
            strand = 0
            ## first lesion at position 1, second lesion at position 2
            idx1 = 1; idx2 = 2
            ## First lesion repaired by MMR, which strips out bases between 0 and 3
            exo_lo = 0; exo_hi = 3
            r1 = Repair(strand = strand, idx = idx1,
                        aid_time = aid_time_1, repair_time = repair_time_1,
                        repair_type = "mmr", exo_lo = exo_lo, exo_hi = exo_hi)
            r2 = Repair(strand = strand, idx = idx2,
                        aid_time = aid_time_2, repair_time = repair_time_2,
                        repair_type = "ber", exo_lo = np.nan, exo_hi = np.nan)
            self.mp.repair_types = [r1, r2]
            ## check that after the first repair is processed, the second gets set to "none"
            self.mp.process_one_repair(r1, self.mp.start_seq)
            return self.mp.repair_types[1].repair_type
        ## the first lesion is stripped out before it has a chance to be repaired
        self.assertEqual(repair_processing_results(1, 2, 3, 4), "none")
        ## first lesion repaired but the C base doesn't change, so the
        ## second lesion is still valid
        self.assertEqual(repair_processing_results(1, 2, 1.5, 4), "ber")

class testNucleotideSampling(unittest.TestCase):

    def setUp(self):
        ## set up a mp with deterministic ber and pol eta parameters
        naive_seq = Seq("CGCA", alphabet=IUPAC.unambiguous_dna)
        cm = aid_context_model = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
        ## BER always changes C to A
        ber_params=[1, 0, 0, 0]
        ## here pol eta always changes C to T
        pol_eta_params={
            'A': [1, 0, 0, 0],
            'G': [0, 1, 0, 0],
            'C': [0, 0, 0, 1],
            'T': [0, 0, 0, 1]
        }

        self.mp = MutationProcess(
            naive_seq,
            pol_eta_params = pol_eta_params,
            ber_params = ber_params,
            aid_context_model = cm)


    def test_sample_ber(self):
        ## Set up with deterministic set of pol eta params, check that
        ## it gets the right one
        self.assertEqual(self.mp.sample_ber(), "A")

    def test_sample_pol_eta(self):
        ## Set up with deterministic set of pol eta params, check that
        ## it gets the right one
        self.assertEqual(self.mp.sample_pol_eta("A"), "A")
        self.assertEqual(self.mp.sample_pol_eta("T"), "T")
        self.assertEqual(self.mp.sample_pol_eta("C"), "T")
        self.assertEqual(self.mp.sample_pol_eta("G"), "G")

    def test_sample_repaired_sequence(self):
        ## Set up a deterministic mp, check that the sequence comes
        ## out correctly
        pass

    def test_make_complement(self):
        self.assertEqual(make_complement(["A", "T", "U", "G", "C"]), ["T", "A", "A", "C", "G"])



class testLesionCreation(unittest.TestCase):

    def setUp(self):
        naive_seq = Seq("AGCA", alphabet=IUPAC.unambiguous_dna)
        cm = aid_context_model = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
        self.mp = MutationProcess(
            naive_seq,
            aid_context_model = cm)

    def test_make_aid_lesions(self):
        lesions = make_aid_lesions(self.mp.start_seq, self.mp.aid_context_model, self.mp.gp_lengthscale, self.mp.overall_rate)
        ## Correct shape
        self.assertEqual(lesions.shape[1], 3)
        ## All positive values
        self.assertTrue((lesions >= 0).all())
        ## Lesions only at C bases
        for i in range(lesions.shape[0]):
            self.assertEqual(self.mp.start_seq[lesions[i,0],lesions[i,1]], "C")

    def test_draw_from_gp(self):
        base_rate_array = make_base_rate_array(self.mp.start_seq,
                                               self.mp.aid_context_model,
                                               overall_rate = 10)
        pp_draw = draw_poisson_process_from_base_rate(base_rate_array)
        gp_draw = draw_from_gp(input_points = pp_draw, gp_lengthscale = self.mp.gp_lengthscale)
        ## Correct shape
        self.assertEqual(len(gp_draw), pp_draw.shape[0])

    def test_draw_from_poisson_base_rate(self):
        base_rate_array = make_base_rate_array(self.mp.start_seq,
                                               self.mp.aid_context_model,
                                               overall_rate = 10)
        pp_draw = draw_poisson_process_from_base_rate(base_rate_array)
        self.assertEqual(pp_draw.shape[1], 3)
        self.assertTrue((pp_draw[:,2] >= 0).all())
        self.assertTrue((pp_draw[:,2] < 1).all())
        self.assertTrue((pp_draw[:,1] < self.mp.seq_len).all())
        self.assertTrue((pp_draw[:,0] <= 1).all())
        self.assertTrue((pp_draw[:,0] >= 0).all())

    def test_make_base_rate_array(self):
        n_time_bins = 100
        base_rate_array = make_base_rate_array(self.mp.start_seq,
                                               self.mp.aid_context_model,
                                               overall_rate = self.mp.overall_rate)
        ## correct shape
        self.assertEqual(base_rate_array.shape, (2, self.mp.seq_len))
        ## zero rates at non-C bases
        for index, rate in np.ndenumerate(base_rate_array):
            if self.mp.start_seq[index[0],index[1]] != "C":
                self.assertEqual(base_rate_array[index[0],index[1]], 0)

# class testAIDLesion(unittest.TestCase):

#     def setUp(self):
#         pass

#     def test_aid_lesion(self):
#         cm = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
#         seq = Seq("AGCT", alphabet=IUPAC.unambiguous_dna)
#         (lesions_fw, lesions_rc) = make_aid_lesions(seq, cm)
#         # all the lesions should be at C positions
#         self.assertTrue(all([seq[i] == "C" for i in lesions_fw]))
#         self.assertTrue(all([seq.reverse_complement()[i] == "C"
#                              for i in lesions_rc]))

#     def test_lesions(self):
#         cm = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
#         s = Seq("C" * 10 + "A" * 10 + "AACAGCAGCGACGTC",
#                 alphabet=IUPAC.unambiguous_dna)
#         (lesions_fw, lesions_rc) = make_aid_lesions(s, cm, 10, 5)

#     def test_get_context(self):
#         cm = ContextModel(context_length=5, pos_mutating=2, csv_string=None)
#         cm2 = ContextModel(context_length=3, pos_mutating=2, csv_string=None)
#         s = "AAGCT"
#         self.assertEqual(cm.get_context(idx=2, sequence=s), "AAGCT")
#         self.assertEqual(cm2.get_context(idx=2, sequence=s), "AAG")
#         self.assertEqual(cm.get_context(idx=0, sequence=s), "NNAAG")
#         self.assertEqual(cm.get_context(idx=4, sequence=s), "GCTNN")

#     def test_in_flank(self):
#         cm = ContextModel(context_length=5, pos_mutating=2, csv_string=None)
#         self.assertTrue(cm.in_flank(idx=0, seq_len=10))
#         self.assertTrue(cm.in_flank(idx=1, seq_len=10))
#         self.assertFalse(cm.in_flank(idx=2, seq_len=10))
#         self.assertTrue(cm.in_flank(idx=9, seq_len=10))
#         self.assertTrue(cm.in_flank(idx=8, seq_len=10))
#         self.assertFalse(cm.in_flank(idx=7, seq_len=10))
#         cm2 = ContextModel(context_length=3, pos_mutating=2, csv_string=None)
#         self.assertTrue(cm2.in_flank(idx=0, seq_len=10))
#         self.assertTrue(cm2.in_flank(idx=1, seq_len=10))
#         self.assertFalse(cm2.in_flank(idx=2, seq_len=10))
#         self.assertFalse(cm2.in_flank(idx=9, seq_len=10))
#         self.assertFalse(cm2.in_flank(idx=8, seq_len=10))
#         self.assertFalse(cm2.in_flank(idx=7, seq_len=10))

#     def test_compute_marginal_prob(self):
#         cm = ContextModel(context_length=3, pos_mutating=2, csv_string=None)
#         cm.context_dict = {}
#         cm.context_dict["AAC"] = .5
#         cm.context_dict["AGC"] = .1
#         self.assertEqual(cm.compute_marginal_prob("NAC"), .5)
#         self.assertEqual(cm.compute_marginal_prob("NNC"), .3)


if __name__ == '__main__':
    unittest.main()
