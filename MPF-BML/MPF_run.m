function [JL1L2_allcouplings]= MPF_run(msa_bin_unique,weight_seq_unique,num_mutants_combine_array,phi_opt,options_MPF)
% MPF_run(msa_bin_unique,weight_seq_unique,num_mutants_combine_array,phi_opt,options_MPF)
% 
% Run MPF using extended binary matrix

% Inputs:
%       msa_bin_unique - extended unique binary matrix
%       weight_seq_unique -  sequence weighting of sequences in msa_bin_unique
%       num_mutants_combine_array - number of mutants per residue
%       phi_opt -  number of patients
%       options_MPF -  number of patients
%
% Outputs:
%       JL1L2_allcouplings - fields/coupling matrix
%        
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% Process Options
if nargin < 5
    options_MPF = [];
end

num_patients = sum(weight_seq_unique);


[options_MPF.verbose,options_MPF.opt_tol,options_MPF.prog_tol,options_MPF.max_iter,options_MPF.suffDec,options_MPF.memory,options_MPF.lambda_h,options_MPF.lambda_J,options_MPF.gamma_h,options_MPF.gamma_J,max_iter_MPF] = ...
    myProcessOptions(options_MPF,'verbose',1,'opt_tol',1e-20,'prog_tol',1e-20,...
    'max_iter',10000,'suffDec',1e-4,'memory',10,'lambda_h',0,'lambda_J',10/num_patients,'gamma_h',0,'gamma_J',10/num_patients,'max_iter_MPF',2);

% Store location of ones in msa_bin_unique, and the sequences which are one
% Hamming distance away. Used for faster computation of MPF algorithm
[xstartpos darraystartNonzeroPos darrayNonzeroPos darrayNonzero dactvalues xarray xpos num_mutant_xarray num_mutant_dplusxarray] = helper_nonzero_entries(num_mutants_combine_array, msa_bin_unique);

mut_mat = full(((msa_bin_unique')*diag(weight_seq_unique)*msa_bin_unique))/num_patients; % mutant probability matrix

num_residues_binary = size(msa_bin_unique,2);
cumul_num_mutants_combine_array = cumsum(num_mutants_combine_array);

% Initialize to fields-only model with small couplings
single_mut = diag(mut_mat);
phi_temp = [0 cumul_num_mutants_combine_array];
single_mut_norm=[];
for indCount=1:length(cumul_num_mutants_combine_array)
    curr_sites{indCount} = phi_temp(indCount)+1:phi_temp(indCount+1);
    muts = single_mut(curr_sites{indCount})';
    single_mut_norm = [single_mut_norm muts/(1-sum(muts))];
end

fields_landscape = diag(-log(single_mut_norm));

J_MINFLOW_array_init = 0.01*ones(num_residues_binary,num_residues_binary);
J_MINFLOW_array_init = J_MINFLOW_array_init-diag(diag(J_MINFLOW_array_init));
J_MINFLOW_array_init = J_MINFLOW_array_init + fields_landscape;

diagone = diag(ones(1,num_residues_binary));
ind_diag = find(diagone(:)==1); % location of diagonal entries in flattened array
ind_nodiag = find(diagone(:)==0); % location of non-diagonal entries in flattend array

for indMPF=1:max_iter_MPF
        J_MINFLOW_array_init = helper_L1(@(J)helper_MPF_run( J, msa_bin_unique',xstartpos,darraystartNonzeroPos,darrayNonzeroPos,darrayNonzero,dactvalues,xarray,xpos,weight_seq_unique,num_mutant_xarray,num_mutant_dplusxarray  ),J_MINFLOW_array_init(:),ind_diag,ind_nodiag,options_MPF);
end
JL1L2_allcouplings = make_symmetric(J_MINFLOW_array_init);
JL1L2_allcouplings = reshape(JL1L2_allcouplings,num_residues_binary,num_residues_binary);
JL1L2_allcouplings = triu(JL1L2_allcouplings)*2 - diag(diag(JL1L2_allcouplings));
