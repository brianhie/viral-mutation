function [phi_opt, beta_mean] = mutant_combining(msa_aa,varargin)
% mutantCombining(msa_aa,weight_seq,phi_array)
% Calculates the optimal mutant combining factor
%
% Inputs:
%       msa_aa - matrix of characters (aka amino acid MSA). Rows correspond to
%                sequences, and colums to observed states at a particular residue
%       mutantCombining(...,'weight_seq',weight_seq)
%                     weight of each sequence, length is equal to the the number of
%                     rows in msa_aa. Default is equal weighting
%       mutantCombining(...,'phi_array',phi_array) 
%                     vector of combining factors to sweep over
%                    Default=[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9:0.1:1]
%
% Outputs:
%       phi_opt - optimal entropy combining factor from elements in phi_array
%                 based on
%       beta_mean - mean beta values for each element in phi_array
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

num_seq = size(msa_aa,1);

% set arguments
if  nargin > 1
    if rem(nargin,2) ~= 1
        error('Incorrect number of arguments.');
    end
    okargs = {'weight_seq','phi_array'};
    for j=1:2:nargin-1
        pname = varargin{j};
        pval = varargin{j+1};
        k = find(strncmpi(pname,okargs,numel(pname)));
        if isempty(k)
            error(strcat('Unknown argument', pname));
        else
            switch(k)
                case 1  
                    weight_seq = pval;
                case 2 
                    phi_array=pval;
                    
            end
        end
    end
end

% Set default values
if ~exist('weight_seq')
  weight_seq = ones(num_seq,1);% equal weighting
end

if ~exist('phi_array')
    phi_array=[0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9:0.1:1];
end

[freq_single_array,amino_single_array] = helper_single_mut(msa_aa,weight_seq);

protein_length_aa = length(freq_single_array);
num_patients = sum(weight_seq);

beta_mat = zeros(length(phi_array),protein_length_aa); % beta values for each phi and residue
for ind_res = 1:protein_length_aa
    curr_freq = freq_single_array{ind_res};
    curr_num_amino = length(curr_freq); % number of amino acids in residue ind_res
    
    % entropy
    entropy_all = -sum(curr_freq.*log(curr_freq));
    
    % ratio of combined mutant entropy over entropy for different ki
    entropy_ratio = zeros(curr_num_amino-1,1);
    for ki=1:curr_num_amino-1
        fbar = sum(curr_freq(ki+1:end)); % frequency of all amino acids
        entropy_ki = -sum(curr_freq(1:ki).*log(curr_freq(1:ki))) - fbar*log(fbar);
        entropy_ratio(ki) = entropy_ki/entropy_all;
    end
    
    % calculate beta
    for ind_phi = 1:length(phi_array)
        
        phi=phi_array(ind_phi);
        ki_phi = find(entropy_ratio>=phi); %ki corresponding to phi
        if length(ki_phi>0)
            ki_phi = ki_phi(1);
            fbar2 = zeros(1,curr_num_amino);
            fbar2(1:ki_phi) = curr_freq(1:ki_phi);
            fbar2(ki_phi+1) = sum(curr_freq(ki_phi+1:end));
            
            beta_mat(ind_phi,ind_res) = sum((curr_freq(2:end)-fbar2(2:end)).^2)/sum(curr_freq(2:end).*(1-curr_freq(2:end))/num_patients);
        else % conserved site
            beta_mat(ind_phi,ind_res)=0;
        end
    end
    
end

beta_mean = mean(beta_mat');

% find the phi with the mean of beta closest to one
[min_value min_ind] = min(abs(beta_mean-1));
phi_opt = phi_array(min_ind);
