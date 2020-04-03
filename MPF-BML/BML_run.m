function J_MINFLOW_mat = BML_run(J_MINFLOW_mat,msa_bin_unique,weight_seq_unique,num_mutants_combine_array,options_BML)
% BML_run(J_MINFLOW_mat_array,num_mutants_combine_array,msa_bin_unique,weight_seq_unique,options_BML)
%
% Implementation of RPROP algorithm
%
% Inputs:
%       J_MINFLOW_mat_array - initial field/couplings
%       msa_bin_unique - extended unique binary matrix
%       weight_seq_unique -  sequence weighting of sequences in msa_bin_unique
%       num_mutants_combine_array - number of mutants per residue
%       options_BML -  options
%
% Outputs:
%       J_MINFLOW_mat_array - fields/coupling matrix from BML
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J_MINFLOW_mat_array = ((J_MINFLOW_mat + J_MINFLOW_mat')/2);
J_MINFLOW_mat_array = J_MINFLOW_mat_array(:);

%% Process Options
if nargin < 5
    options_BML = [];
end



[verbose,eps_max,optTol,progTol,no_iterations,no_seeds,par_opt,thin,burnin,...
    no_sample_MCMC,gammaJ_const,gammaH_const,hpos,Jpos,hneg,Jneg,hdeltamax,...
    Jdeltamax,hdeltamin,Jdeltamin] = ...
    myProcessOptions(options_BML,'verbose',1,'eps_max',1,'optTol',1e-20,'progTol',1e-20,...
    'no_iterations',10000,'no_seeds',1,'par_opt',0,'thin',3e3,'burnin',1e4,...
    'no_sample_MCMC',1e7,'gammaJ_const',0.0001,'gammaH_const',0.0001,'hpos',1.05,...
    'Jpos',1.05,'hneg',0.95,'Jneg',0.95,'hdeltamax',0.00000001,...
    'Jdeltamax',0.0001,'hdeltamin',0.00000001,'Jdeltamin',0.00000001);

if par_opt
    poolobj = gcp('nocreate');
    delete(poolobj);
    parpool(no_seeds)
end

num_patients = sum(weight_seq_unique);
mut_mat = full(((msa_bin_unique')*diag(weight_seq_unique)*msa_bin_unique))/num_patients; % mutant probability matrix
[delta_cij delta_cij_bound] = helper_covar(mut_mat,num_patients);

% Helper variables used in the algorithm
cumul_num_mutants_combine_array = cumsum(num_mutants_combine_array);
num_residues_binary = size(msa_bin_unique,2);
protein_length_aa=length(num_mutants_combine_array);
msa_bin_unique = full(msa_bin_unique);

% Initialize parameterse
J_MINFLOW_mat = reshape(J_MINFLOW_mat_array,num_residues_binary,num_residues_binary);
h_curr = diag(J_MINFLOW_mat);
J_curr = J_MINFLOW_mat;

% Obtain the location of MCMC seeds
randsample_number = no_seeds;
randvalue = randi([1 size(msa_bin_unique,1)],1, randsample_number);

% Obtain the probability of number of mutations
pdf_MSA = helper_PN(weight_seq_unique,msa_bin_unique);
max_mutant = length(pdf_MSA);

% RPROP parameters

dJprev = zeros(1,num_residues_binary*num_residues_binary);
dHprev = zeros(1,num_residues_binary);

gammaJ=gammaJ_const*ones(1,num_residues_binary*num_residues_binary);
gammaH=gammaH_const*ones(1,num_residues_binary);

% gammaJ=gammaJ_const*ones(1,num_residues_binary*num_residues_binary);
% gammaH=gammaH_const*ones(1,num_residues_binary);
%
% hpos = 1.05;
% Jpos= 1.05;
%
% hneg=0.95;
% Jneg=0.95;
%
% hdeltamax = 0.001;
% Jdeltamax = 0.0001;
% hdeltamin = 0.00000001;
% Jdeltamin = 0.00000001;

maxNumber=99999999999999999;
hlower=-maxNumber*ones(1,num_residues_binary);
hhigher = maxNumber*ones(1,num_residues_binary);
Jlower = -maxNumber*ones(1,num_residues_binary*num_residues_binary);
Jhigher = maxNumber*ones(1,num_residues_binary*num_residues_binary);

% number of final MCMC samples after burning and thinning
number_samples = ceil((no_sample_MCMC-burnin)/thin);

% Find the location of zero couplings. These will be manually
% set to zero in the algorithm.
indZero = find(J_MINFLOW_mat_array==0);

for ite2=1:no_iterations
    time_MCMC = tic();
    
    J_MINFLOW_mat = (J_curr - diag(diag(J_curr))) + diag(h_curr);
    J_MINFLOW_mat = (J_MINFLOW_mat+J_MINFLOW_mat')/2;
    J_MINFLOW_mat_array = J_MINFLOW_mat(:);
    J_MINFLOW_mat_array(indZero)=0;
    
    totalnosample_par = zeros(1,no_seeds);
    double_mutantsum_par = zeros(no_seeds,num_residues_binary*num_residues_binary);
    num_mutant_array_par = zeros(no_seeds,number_samples);
    
    if par_opt
        parfor ite_seeds=1:no_seeds
            
            curr_vector = msa_bin_unique(randvalue(ite_seeds),:);
            
            [doublemutant nosample number_mutants]= gibbs_potts_mex(curr_vector,J_MINFLOW_mat_array,num_residues_binary,no_sample_MCMC,cumul_num_mutants_combine_array,num_mutants_combine_array,burnin,thin,number_samples,protein_length_aa);
            double_mutantsum_par(ite_seeds,:) =doublemutant;
            totalnosample_par(ite_seeds)= nosample;
            num_mutant_array_par(ite_seeds,:) =number_mutants;
            
            
        end
    else
        for ite_seeds=1:no_seeds
            
            curr_vector = msa_bin_unique(randvalue(ite_seeds),:);
            
            [doublemutant nosample number_mutants]= gibbs_potts_mex(curr_vector,J_MINFLOW_mat_array,num_residues_binary,no_sample_MCMC,cumul_num_mutants_combine_array,num_mutants_combine_array,burnin,thin,number_samples,protein_length_aa);
            
            double_mutantsum_par(ite_seeds,:) =doublemutant;
            totalnosample_par(ite_seeds)= nosample;
            num_mutant_array_par(ite_seeds,:) =number_mutants;
            
        end
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
    
    
    
    % calcualte gradients
    
    single_mut_MCMC = diag(mut_mat_MCMC);
    single_mut = diag(mut_mat);
    
    dJ = (mut_mat - mut_mat_MCMC);
    dH = (single_mut - single_mut_MCMC)';
    
    
    dJflat = dJ(:)';
    Jcurrflat = J_curr(:);
    
    %%%%%%%%%%%%%%%
    
    htest = dH.*dHprev;
    jtest = dJflat.*dJprev;
    
    indh_pos = find(htest>0);
    indj_pos = find(jtest>0);
    
    indh_neg = find(htest<0);
    indj_neg = find(jtest<0);
    
    indh_eq = find(htest==0);
    indj_eq = find(jtest==0);
    
    gammaH(indh_pos) = min(hpos*gammaH(indh_pos),hdeltamax);
    gammaJ(indj_pos)  = min(Jpos*gammaJ(indj_pos),Jdeltamax);
    
    gammaH(indh_neg) = max(hneg*gammaH(indh_neg),hdeltamin);
    gammaJ(indj_neg)  = max(Jneg*gammaJ(indj_neg),Jdeltamin);
    
    dH(indh_neg)  = 0;
    dJ(indj_neg)=0;
    
    %%%%%%%%%%%%%%%%%%
    % update
    
    dJflattemp = dJ(:);
    
    indh_pos2 = find( dH>0);
    indj_pos2 = find(dJflattemp>0);
    
    indh_neg2 = find( dH<0);
    indj_neg2 = find(dJflattemp<0);
    
    if (ite2>1)
        
        hlower(intersect(indh_neg2,indh_pos)) = hprev(intersect(indh_neg2,indh_pos));
        Jlower(intersect(indj_neg2,indj_pos)) =   Jprev(intersect(indj_neg2,indj_pos));
        
        hhigher(intersect(indh_pos2,indh_pos)) = hprev(intersect(indh_pos2,indh_pos));
        Jhigher(intersect(indj_pos2,indj_pos)) =   Jprev(intersect(indj_pos2,indj_pos));
    end
    
    
    %%%%%%%%%%
    
    cross_thres=1/num_patients;
    [single_error double_error epsilon_max covar_error] = helper_eps(mut_mat,mut_mat_MCMC,delta_cij,delta_cij_bound,num_patients,cumul_num_mutants_combine_array,cross_thres);
    
    
    time_MCMC = toc(time_MCMC);
    if ite2>1
        diffParam = sum(sum(abs(Jprev-Jcurrflat) + sum(abs(hprev-h_curr))));
        diffParamDerivative = sum(sum(abs(dJprev-dJflat) + sum(abs(dHprev-dH))));
        if verbose
            fprintf( 'Iteration %f: Average Eps Single/Double/Connected: %f/%f/%f. Diff Param/Derivative: %f/%f, Time: %f\n', ite2, single_error,double_error,covar_error,diffParam,diffParamDerivative,time_MCMC );
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%5
    % Check termination conditions
    
    if ite2>1
        if diffParam<optTol
            if verbose
                fprintf('Progress in parameters  below progTol\n');
            end
            break;
        end
        
        if diffParamDerivative<progTol
            if verbose
                fprintf('Directional derivative below progTol\n');
            end
            break;
        end
    end
    
    
    if single_error<eps_max & double_error<eps_max
        fprintf('Average epsilon < %f \n',eps_max);
        break;
        
    end
    
    %%%%%%%%%
    
    hprev  = h_curr;
    Jprev = Jcurrflat;
    
    h_curr(intersect(indh_pos2,indh_pos)) = h_curr(intersect(indh_pos2,indh_pos)) - gammaH(intersect(indh_pos2,indh_pos))';
    Jcurrflat(intersect(indj_pos2,indj_pos)) = Jcurrflat(intersect(indj_pos2,indj_pos)) - gammaJ(intersect(indj_pos2,indj_pos))';
    
    h_curr(intersect(indh_neg2,indh_pos)) = h_curr(intersect(indh_neg2,indh_pos)) + gammaH(intersect(indh_neg2,indh_pos))';
    Jcurrflat(intersect(indj_neg2,indj_pos)) = Jcurrflat(intersect(indj_neg2,indj_pos)) + gammaJ(intersect(indj_neg2,indj_pos))';
    
    h_curr(intersect(indh_pos2,indh_eq)) = h_curr(intersect(indh_pos2,indh_eq)) - gammaH(intersect(indh_pos2,indh_eq))';
    Jcurrflat(intersect(indj_pos2,indj_eq)) = Jcurrflat(intersect(indj_pos2,indj_eq)) - gammaJ(intersect(indj_pos2,indj_eq))';
    
    h_curr(intersect(indh_neg2,indh_eq)) = h_curr(intersect(indh_neg2,indh_eq)) + gammaH(intersect(indh_neg2,indh_eq))';
    Jcurrflat(intersect(indj_neg2,indj_eq)) = Jcurrflat(intersect(indj_neg2,indj_eq)) + gammaJ(intersect(indj_neg2,indj_eq))';
    
    J_curr = reshape(Jcurrflat,num_residues_binary,num_residues_binary);
    
    dHprev = dH;
    dJprev = dJflat;
        
    
end
J_MINFLOW_mat = (J_curr - diag(diag(J_curr))) + diag(h_curr);
J_MINFLOW_mat = (J_MINFLOW_mat+J_MINFLOW_mat')/2;
J_MINFLOW_mat_array = J_MINFLOW_mat(:);
J_MINFLOW_mat_array(indZero)=0;
J_MINFLOW_mat = reshape(J_MINFLOW_mat_array,num_residues_binary,num_residues_binary);

if par_opt
    delete(poolobj)
end
