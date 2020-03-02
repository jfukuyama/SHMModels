import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import IUPAC
import csv
import sys
import argparse


def mutation_subset(naive_seq, mutated_seq, base):
    """Pulls out subsets of mutations for use in samm logistic regression.

    Keyword arguments:
    naive_seq -- The germline sequence, as a Seq object.
    mutated_seq -- The mutated sequence, as a Seq object.
    base -- The germline base for the mutations.

    Returns:
    A pair with the naive sequence and a sequence with the relevant subset of mutations.
    """
    if not isinstance(naive_seq, Seq):
        raise TypeError("The naive sequence must be a Seq object")
    if not isinstance(mutated_seq, Seq):
        raise TypeError("The mutated sequence emust be a Seq object")
    if not isinstance(naive_seq.alphabet, type(IUPAC.ambiguous_dna)):
        raise TypeError("the alphabet must be IUPAC.ambiguous_dna")
    if not isinstance(mutated_seq.alphabet, type(IUPAC.ambiguous_dna)):
        raise TypeError("the alphabet must be IUPAC.ambiguous_dna")
    if len(naive_seq) != len(mutated_seq):
        raise ValueError("The naive and mutated sequences must have the same length.")
    mutated_seq_subset = list(str(naive_seq))

    for (i, (b1, b2)) in enumerate(zip(naive_seq, mutated_seq)):
        if (b1 != b2) and (b1 == base):
            mutated_seq_subset[i] = b2
    mutated_seq_subset = Seq("".join(mutated_seq_subset), alphabet=IUPAC.ambiguous_dna)
    return (naive_seq, mutated_seq_subset)


def mutation_subset_from_samm(seqs_file, genes_file, base, strand, output_seqs=None, output_genes=None):
    """Takes files containing mutated sequneces and germline sequences as input and writes a csv containing mutated sequences with a subset of the mutations.

    Keyword arguments:

    seqs_file -- A sequence csv as created by preprocess_data.py in
    samm. Column names germline_name, sequence_name, sequence, group,
    species, locus.
    genes_file -- A genes csv as created by preprocess_data.py in
    samm. Column names germline_name, germline_sequence.
    base -- The base mutated away from. All mutations except for those
    originating from 'base' are reverted to the germline sequence.

    strand -- The strand for the mutation, 'fw', 'rc', or 'both'. If
    'rc', we take the reverse complement of the sequences, and the
    output files for the mutated sequences and the genes are written
    as the reverse complement. If both, we write one set of reverted
    sequences for the forward strand and one set for the reverse
    complementary strand.

    output_file -- The file to write the results to. Writes nothing if
    None.

    Returns: Writes a csv with one column for the naive sequence and
    one column for the sequences with a subset of the initial
    mutations and returns the data frame.

    """
    seqs = pd.read_csv(seqs_file)
    with open(genes_file, mode='r') as infile:
        reader = csv.reader(infile)
        genes_dict = {rows[0]: rows[1] for rows in reader}
        # if we are using the reverse complement or both, we need to
        # add reverse complement germlines
        if strand == 'rc' or strand == 'both':
            for k in genes_dict.keys():
                rc_names = k + 'rc'
                rc_seq = Seq(genes_dict[k], alphabet=IUPAC.unambiguous_dna).reverse_complement()
                genes_dict[rc_names] = str(rc_seq)

    # a list to store the rows of the data frame in
    mutation_subset_rows = []
    # step through the seqs data frame row by row
    for (index, row) in seqs.iterrows():
        mutated_seq = Seq(row["sequence"], alphabet=IUPAC.ambiguous_dna)
        naive_seq = Seq(genes_dict[row["germline_name"]], alphabet=IUPAC.ambiguous_dna)
        if len(mutated_seq) != len(naive_seq):
            print("sequence lengths differ: {} and {}".format(len(mutated_seq), len(naive_seq)))
        if strand == 'fw' or strand == 'both':
            (ms_naive, ms_mut) = mutation_subset(naive_seq, mutated_seq, base)
            new_row = row.copy()
            new_row["sequence"] = str(ms_mut)
            mutation_subset_rows.append(new_row)
        if strand == 'rc' or strand == 'both':
            (ms_naive, ms_mut) = mutation_subset(naive_seq.reverse_complement(), mutated_seq.reverse_complement(), base)
            new_row = row.copy()
            new_row["sequence"] = str(ms_mut)
            new_row["germline_name"] = row["germline_name"] + 'rc'
            mutation_subset_rows.append(new_row)

    mutation_df = pd.DataFrame(mutation_subset_rows)
    if output_seqs is not None:
        mutation_df.to_csv(output_seqs, index=False)
    if output_genes is not None:
        # genes file has header germline_name,germline_sequence
        with open(output_genes, 'wb') as f:
            w = csv.writer(f)
            w.writerow(['germline_name', 'germline_sequence'])
            w.writerows(genes_dict.items())

    return mutation_df


def parse_args():
    """parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-seqs",
                        type=str,
                        help="input csv from samm with mutated sequences")
    parser.add_argument("--input-genes",
                        type=str,
                        help="input csv from samm with the gene descriptions")
    parser.add_argument("--base",
                        type=str,
                        help="the germline base mutated away from")
    parser.add_argument("--strand",
                        type=str,
                        help="either 'fw' or 'rc'")
    parser.add_argument("--output-file",
                        type=str,
                        help="where to write the sequences with a subset of mutations")
    args = parser.parse_args()
    return args

def main(args=sys.argv[1:]):
    args = parse_args()
    mutation_subset_from_samm(args.input_seqs, args.input_genes, args.base.lower(), args.strand, args.output_file)


if __name__ == "__main__":
    main(sys.argv[1:])
