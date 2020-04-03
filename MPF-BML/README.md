## Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Required Toolboxes](#required-toolboxes)
* [Usage of the MPF-BML code](#usage-of-the-mpf-bml-code)
  * [Step 1: Mutant Combining](#step-1-mutant-combining)
    * [Example usage](#example-usage)
  * [Intermediate step: helper variables](#intermediate-step-helper-variables)
    * [Example usage](#example-usage-1)
  * [Step 2: MPF](#step-2-mpf)
    * [Example usage](#example-usage-2)
  * [Step 3: BML](#step-3-bml)
    * [Example usage](#example-usage-3)
* [gp160 processed MSA](#gp160-processed-msa)
* [gp160 landscape](#gp160-landscape)
* [Troubleshooting](#troubleshooting)


## Overview

This repository contains 

1. Implementation (in MATLAB) of the Minimum Probability Flow-Boltzmann Machine Learning (MPF-BML) framework, an algorithm which infers the parameters of the maximum entropy distribution (Boltzmann distribution) using  the Potts model.  We apply this framework  to infer the fitness landscape of gp160 based on its sequence data, however the framework can be used for any problem requiring inference of the maximum entropy parameters. (gp160 is a protein in HIV which is the primary target of antibodies.)
2. Preprocessed mulitple sequence alignment (MSA) of gp160 (FASTA format) and
3. Gp160 landscape (MATLAB .mat format)

as described in 

RHY Louie, KJ Kaczorowski, JP Barton, AK Chakraborty, MR McKay, "The fitness landscape of the Human Immunodeficiency Virus envelope protein that is targeted by antibodies", Proceedings of the National Academy of Sciences (PNAS), 2018

## Installation

To run the MPF and BML components of the framework, there are two C MEX files in the "Helper Functions" folder which need to be built. This can be compiled by the following instructions:

1. Open MATLAB
2. Change directory to the "Helper Functions" folder.
3. In the command prompt, enter` mex K_dK_MPF.c`
4. In the command prompt, enter  `mex gibbs_potts_mex.c`

## Required Toolboxes

The following MATLAB toolboxes are required:

1. Bioinformatics
2. Communications System
3. Parallel Computing

## Usage of the MPF-BML code

The MPF-BML computational framework is an algorithm to infer the field and coupling parameters of the Maximum Entropy distribution.  An example working code can be found in the script

`main_MPF_BML.m`

which runs the complete framework, and plots various statistics to confirm the inferred parameters. The code has been deliberately left as a script, not a function, to allow users  to explore the different steps of the framework. Example data is provided. 

The framework comprises of three main functions corresponding to the three key steps of the algorithm, each of which can be run independently of the other. Before describing the steps, first note that each function requires as input  a sample character matrix `msa_aa`, which can be formed from a FASTA file with name `fasta_name` by

```
[Header_fasta, Sequence_fasta] = fastaread(fasta_name);
msa_aa = cell2mat(Sequence_fasta');
```

The above code however, assumes that each sequence is of the same length, which you will have to ensure yourself. An example FASTA file is provided in the folder "MSA and Landscape".

We now describe the three steps, where we assume all examples are to be run in the MATLAB command prompt.

#### Step 1: Mutant Combining

The purpose of this step is to reduce the number of states (resulting in a decrease in the number of couplings)  to achieve a balance between bias and variance. The function which implements this is `mutantCombining` and the output `phi_opt`  is the optimal combining factor  which represents the fraction of the entropy obtained by "coarse-graining" or combining the least-frequent states to one state, compared to the entropy without combining. Note that `phi_opt=0` corresponds to the pure Ising case, while `phi_opt=1` corresponds to the pure Potts case.

##### Example usage

Choose the optimal combining factor from `phi_array`, a vector of possible values, and `weight_seq`, the weighting per sequence.

```
phi_array = [0:0.1:1]; 
weight_seq = ones(size(msa_aa,1),1) ; % equal weighting per patient
phi_opt = mutant_combining(msa_aa, 'weight_seq',weight_seq,'phi_array',phi_array);
```

The default values of weight_seq is set to equal weighting per patient, i.e.,

`weight_seq = ones(size(msa_aa,1),1) ; % equal weighting per patient `

while the default value of phi_array is

`phi_array=[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9:0.1:1]; `

Thus running 

`phi_opt = mutant_combining(msa_aa);`

will produce the optimal combining factor using these default values.

#### Intermediate step: helper variables

The MPF (Step 2) and BML (Step 3) functions both require  helper variables, produced by the function `helper_variables.m`. These helper variables are 

`msa_bin` - binary extended matrix after combining with factor `phi_opt`

`msa_bin_unique` - unique rows of `msa_bin`

`weight_seq_unique` - weight of each sequence in `msa_bin_unique`

`freq_single_combine_array` - frequency of each amino acid after combining with factor `phi_opt`.

`amino_single_combine_array` - amino acid sorted in decreasing order of frequency after combining with factor `phi_opt`

`num_mutants_combine_array` - number of mutants at each residue

`phi_opt` - optimal combining factor

##### Example usage

Calculate the helper variables for the Ising model and equal weight per patient:

```
phi_opt = 0; % Ising model
weight_seq = ones(size(msa_aa,1),1) ; % equal weighting per patient
[msa_bin, msa_bin_unique,weight_seq_unique,freq_single_combine_array,amino_single_combine_array,num_mutants_combine_array,phi_opt]  = helper_variables(msa_aa,'weight_seq',weight_seq,'phi_opt',phi_opt);
```

If `weight_seq` is not specified, it is set to equal weighting per patient, i.e.,

`weight_seq = ones(size(msa_aa,1),1) ; % equal weighting per patient `

while the default value of `phi_opt` is the Potts model, i.e.,

`phi_opt=1; `

#### Step 2: MPF

This step runs a regularized Potts and mex-function extension of the Minimum-Probability-Flow (MPF) algorithm, as originally proposed in 

Sohl-Dickstein J, Battaglino P, DeWeese MR (2009) Minimum Probability Flow learning. Proc 28th ICML 107(Ml):12.

##### Example usage

`J_MPF = MPF_run(msa_bin_unique,weight_seq_unique,num_mutants_combine_array,phi_opt,options_MPF);`

where the inputs are as described in "Intermediate step: helper variables" above, with the exception of:

`options_MPF` - This is an (optional) options struct file which controls various paramters of the algorithm. The most relevant parameters to tune are the regularization parameters, which can be manually set, e.g., by

```
options_MPF.lambda_J = 10/num_patients; % L1 regularization parameter for the couplings
options_MPF.gamma_J = 10/num_patients; % L2 regularization parameter for the couplings
options_MPF.lambda_h = 0; % L1 regularization parameter for the fields/
options_MPF.gamma_h = 0; % L2 regularization parameter for the fields
```
 
MPF runs a gradient descent algorithm to solve the MPF ojbective function. The tolerance level for a small change in the parameters and the gradient, such that the gradient descent algorithm terminates can be set by (default parameters shown) 

```
options_MPF.opt_tol = 1e-20;
options_MPF.prog_tol = 1e-20;
```

Increasing these values will result in a faster code at the expense of accuracy. Alternatively, the number of iterations in the gradient descent algorithm can be modified by

```
options_MPF.max_iter = 10000;
```

The output is fields/couplings matrix. The energy of sequence `x` can thus be calculated as 

`x'*triu(J_MPF)*x`

#### Step 3: BML

This step implements the RPROP algorithm to  refine the parameters inferred from MPF. RPROP is a gradient descent algorithm which we use to solve the original maximum-likelihood (ML) maximum entropy problem (as described in paper).

##### Example usage

```
J_init = J_MPF(:);
J_MPF_BML =BML_run(J_init,msa_bin_unique,weight_seq_unique,num_mutants_combine_array,options_BML);
```

where the inputs are as described in "Intermediate step: helper variables" above, with the exception of 

`J_init` - A flattened fields/couplings vector which initalizes the BML algorithm.

`options_BML` - An (optional) options struct which controls the RPROP algorithm.  A key difficulty with solving the ML problem is due to the partition function, which renders the gradient difficult to calculate. We thus approximate the gradient using  MCMC simulations. The MCMC parameters used to approximate the gradient can be set, for example, by (default shown)

```
options_BML.thin = 3e3; % thinning parameter
options_BML.burnin = 1e4; % number of samples to burn
options_BML.no_sample_MCMC = 1e7 % number of samples before burning and thinning
```
We also provide the option to run MCMC using multiple seeds using multiple-cores, after which the samples are combined. The relevant options are (default shown)

```
options_BML.par_opt = 0; % using only one core. A value of "1" means multiple-cores are used
options_BML.no_seeds = 1; % number of seeds
```

Finally, the BML algorithm will automatically stop when the average epsilon values are < `epsMax` ( as described in paper), which can be changed by (default shown)

```
options_BML.eps_max = 1; 
```

Thus using the above value, the BML algorithm will terminate when the average epsilon is < 1. 

## gp160 processed MSA

The processed MSA (as described in the paper) in fasta format `hivgp160_processed_MSA.fasta` is in the folder "MSA and Landscape". The weighting of each sequence is in `hivgp160_patient_weighting.mat`.

## gp160 landscape

The gp160 field and coupling (landscape) parameters are in `hivgp160_landscape.mat`in the folder "MSA and Landscape", where `J_MPF_BML` is the field/coupling matrix. The energy of sequence `x` is calculated as 

`x'*triu(J_MPF_BML)*x`

amino_acid_after_combining is the amino acids in decreasing order of frequency at each of the 815 residues, and mut_mat is the mutant probaility matrix (after mutant combining).

## Troubleshooting

##### Which regularization parameter should I choose?

The choice of the regularization used in `MPF_run` should ideally be chosen to achieve a balance between overfitting and underfitting, as described in the paper, which can be achieved through ensuring the "average epsilon" measure is close to one. We recommend that users conduct a sweep of different parameters, and run` BML_run`, whilst keeping an eye out on the average epsilon values (as defined in the paper). These values  are displayed in the command prompt at each iteration, when `BML_run` is run. If the BML is clearly not converging to an epsilon value close to one, then choose another set of regularization parameters. For example, to obtain a rough ballpark of how large the regularization parameters are, try first sweeping the L1 and L2 parameters over [1/10 1 10 100 1000 10000]/sum(weight_seq).

##### The average epsilon is never < 1.

Try changing the regularization parameters, as described above. 

Any other questions or comments, please email raylouie@hotmail.com
