function out = verify_param(Jstore_mat,msa_bin_unique,weight_seq_unique,num_mutants_combine_array)
% verify_param(Jstore,msa_bin_unique,weight_seq_unique,num_mutants_combine_array)
% 
% Verify the couplings

% Inputs:
%       Jstore - fields/couplings
%       msa_bin_unique - extended unique binary matrix
%       weight_seq_unique -  sequence weighting of sequences in msa_bin_unique
%       num_mutants_combine_array - number of mutants per residue
%       options_Verify -  options
%
% Outputs:
%       out - 1
%        
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Jstore = ((Jstore_mat + Jstore_mat')/2);
Jstore = Jstore(:);

%% Process Options
if nargin < 5
    options_Verify = [];
end


[no_seeds,parOpt,thin,burnin,nosim] = ...
    myProcessOptions(options_Verify,'no_seeds',1,'parOpt',0,'thin',3e3,'burnin',1e4,'nosim',1e7);


% out =          helper_verify_MCMCv3(J_MPF,msa_bin_unique,num_mutants_combine_array,thin,burnin,nosim,weight_seq_unique,num_patients,no_seeds);

num_patients = sum(weight_seq_unique);
mut_mat_MSA = full(((msa_bin_unique')*diag(weight_seq_unique)*msa_bin_unique))/num_patients; % mutant probability matrix

[delta_cij delta_cij_bound] = helper_covar(mut_mat_MSA,num_patients);


% \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
msa_bin_unique = full(msa_bin_unique);
num_residues_binary = size(msa_bin_unique,2);
cumul_num_mutants_combine_array  = cumsum(num_mutants_combine_array);
protein_length_aa = length(num_mutants_combine_array);

J_MINFLOW_mat_array = full(Jstore);
J_MINFLOW = reshape(Jstore,num_residues_binary,num_residues_binary);

%%%%%%%%


number_samples = ceil((nosim-burnin)/thin);

totalnosample_par = zeros(1,no_seeds);
double_mutantsum_par = zeros(no_seeds,num_residues_binary*num_residues_binary);
num_mutant_array_par = zeros(no_seeds,number_samples);

randvalue = randi([1 size(msa_bin_unique,1)],1, no_seeds);


t_samp = tic();
% noiterations
for ite_seeds=1:no_seeds
    
    curr_vector = msa_bin_unique(randvalue(ite_seeds),:);
    
    [doublemutant nosample number_mutants]= gibbs_potts_mex(curr_vector,J_MINFLOW_mat_array,num_residues_binary,nosim,cumul_num_mutants_combine_array,num_mutants_combine_array,burnin,thin,number_samples,protein_length_aa);
    
    double_mutantsum_par(ite_seeds,:) =doublemutant;
    totalnosample_par(ite_seeds)= nosample;
    num_mutant_array_par(ite_seeds,:) =number_mutants;
    
    
end

if (no_seeds>1)
    double_mutantsum = sum(double_mutantsum_par);
    totalnosample = sum(totalnosample_par);
    num_mutant_array=num_mutant_array_par(:);
    
    mut_mat_MCMC_array=double_mutantsum/totalnosample;
    mut_mat_MCMC = reshape(mut_mat_MCMC_array,num_residues_binary,num_residues_binary);
    mut_mat_MCMC = (mut_mat_MCMC+mut_mat_MCMC') - diag(diag(mut_mat_MCMC));
    
else
    mut_mat_MCMC_array=double_mutantsum_par/nosample;
    mut_mat_MCMC = reshape(mut_mat_MCMC_array,num_residues_binary,num_residues_binary);
    mut_mat_MCMC = (mut_mat_MCMC+mut_mat_MCMC') - diag(diag(mut_mat_MCMC));
    num_mutant_array=num_mutant_array_par;
end


t_samp = toc(t_samp);
time_MCMC = t_samp;
fprintf( 'MCMC in %f seconds \n', t_samp );

cross_thres=1/num_patients;
[single_error double_error epsilon_max covar_error] = helper_eps(mut_mat_MSA,mut_mat_MCMC,delta_cij,delta_cij_bound,num_patients,cumul_num_mutants_combine_array,cross_thres);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot 1: Single mutant probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

single_MCMC = diag(mut_mat_MCMC);
single_MSA = diag(mut_mat_MSA);

arrayline_min = min(single_MCMC);
arrayline_max = max(single_MCMC);
arrayline = arrayline_min:0.01:arrayline_max;
figure
plot(single_MSA,single_MCMC,'rx');hold;grid off;
plot(arrayline,arrayline,'b')
xlabel('Single mutant probability (MSA)')
ylabel('Single mutant probability (MCMC)')
title(['Average epsilon=' num2str(single_error)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot 2: Double mutant probabilities
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% find location of two points in flattened array
zeros_remove = (ones(num_residues_binary,num_residues_binary));
phi_extend = [1 cumul_num_mutants_combine_array+1];
for aba=1:length(phi_extend)-1
    array_sites = phi_extend(aba):phi_extend(aba+1)-1;
    for indi=1:length(array_sites)
        for indj=1:length(array_sites)
            if (indi~=indj)
                zeros_remove(array_sites(indi),array_sites(indj))=0;
            end
        end
    end
end
zeros_remove = sparse(zeros_remove);

ones_double = sparse(triu(ones(num_residues_binary,num_residues_binary),1));
double_pick_final = zeros_remove.*ones_double;
double_pick_final_flat = sparse(double_pick_final(:));
ind_double_pick_final = find(double_pick_final_flat==1); % location of two points

double_MCMC_flat = mut_mat_MCMC(:);
double_MSA_flat = mut_mat_MSA(:);

double_MCMC_flat = double_MCMC_flat(ind_double_pick_final);
double_MSA_flat = double_MSA_flat(ind_double_pick_final);


arrayline_min = min(double_MCMC_flat);
arrayline_max = max(double_MCMC_flat);
arrayline = arrayline_min:0.01:arrayline_max;
figure
plot(double_MSA_flat,double_MCMC_flat,'rx');hold;grid off;
plot(arrayline,arrayline,'b')
xlabel('Double mutant probability (MSA)')
ylabel('Double mutant probability (MCMC)')
title(['Average epsilon=' num2str(double_error)])


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot 3: Connected correlation (covariance)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

connected_MCMC = mut_mat_MCMC - single_MCMC*single_MCMC';
connected_MSA = mut_mat_MSA - single_MSA*single_MSA';

connected_MCMC_flat = connected_MCMC(:);
connected_MSA_flat = connected_MSA(:);

connected_MCMC_flat = connected_MCMC_flat(ind_double_pick_final);
connected_MSA_flat = connected_MSA_flat(ind_double_pick_final);


arrayline_min = min(connected_MCMC_flat);
arrayline_max = max(connected_MCMC_flat);
arrayline = arrayline_min:0.01:arrayline_max;
figure
plot(connected_MSA_flat,connected_MCMC_flat,'rx');hold;grid off;
plot(arrayline,arrayline,'b')
xlabel('Connected correlation (MSA)')
ylabel('Connected correlation (MCMC)')
title(['Average epsilon=' num2str(covar_error)])

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot 4: Probability of number of mutations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pdf_MSA = helper_PN(weight_seq_unique,msa_bin_unique);
max_mutant = length(pdf_MSA);

for indi=1:max_mutant
    pdf_MCMC(indi) = length(find(num_mutant_array==indi))/length(num_mutant_array);
end

figure
semilogy(1:max_mutant,pdf_MSA,'rx-');hold;grid;
semilogy(1:max_mutant,pdf_MCMC,'bs-')
legend('MSA','MCMC','Location','Best')
xlabel('Number of mutants, x')
ylabel('Frequency of x')


out=1;