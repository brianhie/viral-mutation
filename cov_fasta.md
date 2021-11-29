To run the escape prediction experiments using a FASTA file, you will need two things:

1. A FASTA file containing a single baseline sequence (this is often the "wildtype" sequence). We will call this by its filename, `base_fname`.
2. A FASTA file containing the remaining sequences on which you would like to compute semantic change relative to the baseline sequence. We will call this `target_fname`.

To run the analysis for SARS-CoV-2 Spike, you can run the following command:
```bash
python bin/cov_fasta.py \
    base_fname \
    target_fname \
    --checkpoint models/cov.hdf5 \
    --output results.txt
```
This will output a tab-delimited file with the results of the language model analysis for each sequence in `target_fname`.

We provide example input files in the [`examples/`](examples) directory. The results here were generated with the command:
```bash
python bin/cov_fasta.py \
    examples/example_wt.fa \
    examples/example_target.fa \
    --checkpoint models/cov.hdf5 \
    --output examples/example_results.txt
```

In `examples/example_target.fa`, we see three variants of interest and a "null" distribution of previously surveilled sequences. In all cases, the semantic change is substantially elevated, indicating greater escape potential.