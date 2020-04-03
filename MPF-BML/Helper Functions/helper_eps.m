function [eps_single eps_double eps_max eps_cov] = helper_eps(cross_prod_site_MSA,cross_prod_site_MCMC,delta_cij,delta_cij_bound,numpatients,phi_cumulative,cross_thres)

total_length = size(cross_prod_site_MSA,1);
% calculate covariance matrix for MCMC and MSA samples
single_mut_MCMC = diag(cross_prod_site_MCMC);
single_mut_MCMC_mat = single_mut_MCMC*single_mut_MCMC';
cov_mat_MCMC = cross_prod_site_MCMC - single_mut_MCMC_mat;

single_mut_MSA = diag(cross_prod_site_MSA);
single_mut_MSA_mat = single_mut_MSA*single_mut_MSA';
cov_mat_MSA = cross_prod_site_MSA - single_mut_MSA_mat;

% calculate epsilon in matrix form
error_total = (cross_prod_site_MCMC-cross_prod_site_MSA)./sqrt(cross_prod_site_MSA.*(1-cross_prod_site_MSA)/numpatients);
error_total_bound= (cross_prod_site_MCMC-cross_prod_site_MSA)./sqrt((1/numpatients).*(1-(1/numpatients))/numpatients);
error_total_cov = (cov_mat_MCMC-cov_mat_MSA)./delta_cij;
error_total_cov_bound = (cov_mat_MCMC-cov_mat_MSA)./delta_cij_bound;

error_total_flat = error_total(:);
error_total_bound_flat = error_total_bound(:);
cross_prod_site_flat = cross_prod_site_MSA(:);
cross_prod_site_MCMCflat = cross_prod_site_MCMC(:);
error_total_cov_flat = error_total_cov(:);
error_total_cov_boundflat = error_total_cov_bound(:);

%%%%%%%%%%%%%55



zeros_remove = ones(total_length,total_length);
phi_extend = [1 phi_cumulative+1];
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

%%%%%%%%%%%%%
% calculate single epsilon
ones_single = diag(ones(1,total_length));
ones_single_flat = ones_single(:);
ind_single_pick_final = find(ones_single_flat==1);

cross_prod_site_consider_single = cross_prod_site_flat(ind_single_pick_final);
error_flat_consider_single = error_total_flat(ind_single_pick_final);
error_bound_flat_consider_single = error_total_bound_flat(ind_single_pick_final);
ind_smallvalues_single = find(cross_prod_site_consider_single<=cross_thres);
error_flat_consider_single(ind_smallvalues_single) = error_bound_flat_consider_single(ind_smallvalues_single);
eps_single = sqrt(mean(error_flat_consider_single.^2));

%%%%%%%%%%%%%%%%%%%5
% calculate double epsilon
ones_double = triu(ones(total_length,total_length),1);
double_pick_final = zeros_remove.*ones_double;
double_pick_final_flat = double_pick_final(:);
ind_double_pick_final = find(double_pick_final_flat==1);

cross_prod_site_consider_double = cross_prod_site_flat(ind_double_pick_final);
error_flat_consider_double = error_total_flat(ind_double_pick_final);
error_bound_flat_consider_double = error_total_bound_flat(ind_double_pick_final);

ind_smallvalues_double = find(cross_prod_site_consider_double<=cross_thres);
error_flat_consider_double(ind_smallvalues_double) = error_bound_flat_consider_double(ind_smallvalues_double);
eps_double = sqrt(mean(error_flat_consider_double.^2));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% calculate epsilon max
ones_total = triu(ones(total_length,total_length));
total_pick_final = zeros_remove.*ones_total;
total_pick_final_flat = total_pick_final(:);
ind_total_pick_final = find(total_pick_final_flat==1);

cross_prod_site_consider = cross_prod_site_flat(ind_total_pick_final);
error_total_flat_consider = error_total_flat(ind_total_pick_final);
error_total_bound_flat_consider = error_total_bound_flat(ind_total_pick_final);

ind_smallvalues = find(cross_prod_site_consider<=cross_thres);
error_total_flat_consider(ind_smallvalues) = error_total_bound_flat_consider(ind_smallvalues);

eps_max = max(abs(error_total_flat_consider))/sqrt(2*log(length(error_total_flat_consider)));

%%%%%%%%%%%%%%%%%%%%%%%
% calculate covariance
cross_prod_site_consider_cov = cross_prod_site_flat(ind_double_pick_final);
error_flat_consider_cov = error_total_cov_flat(ind_double_pick_final);
error_bound_flat_consider_cov = error_total_cov_boundflat(ind_double_pick_final);

ind_smallvalues_cov = find(cross_prod_site_consider_cov<=cross_thres);
error_flat_consider_cov(ind_smallvalues_cov) = error_bound_flat_consider_cov(ind_smallvalues_cov);

eps_cov = sqrt(mean(error_flat_consider_cov.^2));
