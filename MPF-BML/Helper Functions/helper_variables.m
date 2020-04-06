function [msa_bin, msa_bin_unique,weight_seq_unique,freq_single_combine_array,amino_single_combine_array,num_mutants_combine_array,phi_opt] = helper_variables(msa_aa,varargin)
% binMatAfterComb(msa_aa,weight_seq,phi_opt)
%
% Calculates the binary matrix and amino acid information after
% mutant comnbining

% Inputs:
%       msa_aa - matrix of characters (aka amino acid MSA). Rows correspond to
%                sequences, and colums to observed states at a particular residue
%       helper_variables(...,'weight_seq',weight_seq)
%                     weight of each sequence, length is equal to the the number of
%                     rows in msa_aa. Default is equal weighting
%       helper_variables(...,'phi_opt',phi_opt)-  (optional)
%                     mutant combining factor. Default: phi_opt=1 (Potts)
%
% Outputs:
%       msa_bin - binary extended matrix after combining with factor phi_opt
%       msa_bin_unique - unique rows of msa_bin
%       weight_seq_unique - weight of each sequence in msa_bin_unique
%       freq_single_combine_array - frequency of each amino acid after
%                         combining with factor phi_opt.
%       amino_single_combin_array - amino acid sorted in decreasing order
%                         of frequency after combining with factor phi_opt
%       num_mutants_combine_array - number of mutants at each residue
%       phi_opt - optimal combining factor
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



num_seq = size(msa_aa,1);

% set arguments
if  nargin > 1
    if rem(nargin,2) ~= 1
        error('Incorrect number of arguments.');
    end
    okargs = {'weight_seq','phi_opt'};
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
                    phi_opt=pval;

            end
        end
    end
end

% Set default values
if ~exist('weight_seq')
  weight_seq = ones(num_seq,1);% equal weighting
end

if ~exist('phi_opt')
    phi_opt=1;
end

[freq_single_array,amino_single_array] = helper_single_mut(msa_aa,weight_seq);

num_seq = size(msa_aa,1);

for aba=1:30
    temp_matrix=[];
    temp_matrix = fliplr(eye(aba));
    temp_matrix = [zeros(1,size(temp_matrix,2)) ; temp_matrix ];
    bin_matrix{aba} =temp_matrix;

end

protein_length_aa = length(freq_single_array);
curr_start_pos = 0;
freq_single_combine_array = cell(protein_length_aa,1);
amino_single_combine_array = cell(protein_length_aa,1);
num_mutants_combine_array = zeros(1,protein_length_aa);

for ind_res = 1:protein_length_aa
    curr_freq = freq_single_array{ind_res};
    curr_amino = amino_single_array{ind_res};
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

    ki_phi = find(entropy_ratio>=phi_opt); %ki corresponding to phi_opt
    if length(ki_phi)==0 % 100% conserved
        ki_phi=0;
    else
        ki_phi = ki_phi(1);
    end
    fbar2 = zeros(1,ki_phi+1);
    fbar2(1:ki_phi) = curr_freq(1:ki_phi);
    fbar2(ki_phi+1) = sum(curr_freq(ki_phi+1:end));
    amino_single_combine_array{ind_res} = curr_amino(1:ki_phi+1);
    num_mutants_combine_array(ind_res) = ki_phi;
    freq_single_combine_array{ind_res} = fbar2;
    freq_mutant_combine_array{ind_res} = fliplr(fbar2(2:end));

    curr_amino_combine= curr_amino(1:ki_phi+1);
    for ind_seq=1:num_seq

        loc_amino = find(msa_aa(ind_seq,ind_res)==curr_amino_combine);
        if (length(loc_amino)==0)
            loc_amino=length(curr_amino_combine);
        end

        curr_bin_matrix = bin_matrix{ki_phi};
        bin_value = curr_bin_matrix(loc_amino,:);

        msa_bin(ind_seq,curr_start_pos+1:curr_start_pos+ki_phi)=bin_value; % binary potts matrix
    end
    curr_start_pos =   curr_start_pos +ki_phi;

end

% form new binary MSA based on unique sequences
[msa_bin_unique ind1 ind2]= unique(msa_bin,'rows');

% find new patient weighting WEIGHT_SEQ_UNIQUE based on unique sequences
for indi_bin = 1:length(ind1)
    num_term = ind2(ind1(indi_bin));
    ind_values = find(ind2==num_term);
    weight_seq_unique(indi_bin) = sum(weight_seq(ind_values));
end
