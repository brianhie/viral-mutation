function [delta_cij delta_cij_bound] = helper_covar(mut_mat,num_patients)

for indi=1:size(mut_mat,1)
    for indj=1:size(mut_mat,2)
        pij = mut_mat(indi,indj);
        pi = mut_mat(indi,indi);
        pj = mut_mat(indj,indj);
        delta_pi_temp = sqrt(pi*(1-pi)/num_patients);
        delta_pj_temp = sqrt(pj*(1-pj)/num_patients);
        delta_pij_temp = sqrt(pij*(1-pij)/num_patients);
        
        delta_cij(indi,indj) = sqrt(var_sample_covar(pi,pj,pij,num_patients));
        delta_cij_bound(indi,indj) = sqrt(var_sample_covar(1/num_patients,1/num_patients,1/num_patients,num_patients));
                
    end
    
end

