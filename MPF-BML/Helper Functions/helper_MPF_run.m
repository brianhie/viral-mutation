function [K, dK] = helper_MPF_run(Jflat, msa_aa,xstartpos,darraystartNonzeroPos,darrayNonzeroPos,darrayNonzero,dactvalues,xarray,xpos,prob_D,num_mutant_xarray,num_mutant_dplusxarray  )
%   helper_MPF_run returns the MPF objective function and gradient
%
%   Inputs:   
%   Jflat: flatted coupling matrix
%   msa_aa: amino acid MSA
%   The rest of the input variables store the location of the mutants, and
%   calculated automatically from the helper_nonzero_entries function

[num_residues_binary, num_sequences_unique] = size( msa_aa );


num_d2 = ones(1,num_sequences_unique)*num_residues_binary;
dist_length = max([num_mutant_xarray num_mutant_dplusxarray]);

J = reshape( Jflat, [num_residues_binary, num_residues_binary] );
J = (J + J')/2;
J = J(:);

% mex function implementing MPF
[K dK] = K_dK_MPF(num_sequences_unique,num_residues_binary,xstartpos,num_d2,darraystartNonzeroPos,darrayNonzeroPos,darrayNonzero,dactvalues,xarray,J,xpos,prob_D,num_mutant_xarray,num_mutant_dplusxarray,dist_length );
dK=dK';
