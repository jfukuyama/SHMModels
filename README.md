# SHMModels

This repo contains functions for simulating from mechanistically-informed models of somatic hypermutation.
The package can be installed using
```
pip install .
```

We can simulate mutated sequences by setting up a "mutation process" object and then calling the `generate_mutations` function.
After mutations are generated, the MutationProcess object will contain the mutated sequence (in `repaired_sequence`) and an array giving the number of AID lesions at each position (`aid_lesions_per_site`).
```
import pkgutil
from Bio.Seq import Seq
from Bio.Alphabet import IUPAC
from SHMModels.mutation_processing import *
from SHMModels.fitted_models import *
from SHMModels.simulate_mutations import *

naive_seq = Seq("CGCA", alphabet=IUPAC.unambiguous_dna)

## ber always changes C to A
ber_params=[1, 0, 0, 0]
## here pol eta always changes C to T
pol_eta_params={
    'A': [1, 0, 0, 0],
    'G': [0, 1, 0, 0],
    'C': [0, 0, 0, 1],
    'T': [0, 0, 0, 1]
}
cm = aid_context_model = ContextModel(3, 2, pkgutil.get_data("SHMModels", "data/aid_goodman.csv"))
mp = MutationProcess(naive_seq,
                     aid_context_model = cm,
                     ber_params = ber_params,
                     pol_eta_params = pol_eta_params,
                     ber_lambda = .0100,
                     mmr_lambda = .0100,
                     overall_rate = 10,
                     show_process = False)
mp.generate_mutations()
print(mp.aid_lesions_per_site)
print(mp.repaired_sequence)
```
