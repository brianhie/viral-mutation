function [freq_single_array,amino_single_array] = helper_single_mut(msa_aa,weight_seq)
%HELPER_SINGLE_MUT returns the single mutant probabilities and corresponding amino aicds
%
%   [FREQ_SINGLE_ARRAY,AMINO_SINGLE_ARRAY]=HELPER_SINGLE_MUT(MSA_AA,WEIGHT_SEQ) 
%   returns the single mutant probabilities FREQ_SINGLE_ARRAY and
%   the corresponding amino acids AMINO_SINGLE_ARRAY based on the amino acid 
%   MSA MSA_AA and sequence weighting WEIGHT_SEQ

protein_length_aa = size(msa_aa,2);
num_seq = size(msa_aa,1);
num_patients = sum(weight_seq);

% calculate the single mutant probabilities
freq_single_array = cell(protein_length_aa,1); % frequency of amino acids in descending order
amino_single_array = cell(protein_length_aa,1); % amino acids in descending order of frequency
for ind_res=1:protein_length_aa
    amino_all = msa_aa(:,ind_res); % amino acids at residue ind_res
    amino_all_unique = unique(amino_all); % unique amino acids at residue ind_res
    
    freq_single = zeros(1,length(amino_all_unique));
    for ind_amino=1:length(amino_all_unique)
        curr_amino = amino_all_unique(ind_amino);
        loc_amino = find(amino_all==curr_amino);
        
        % amino_one: array indicating sequence location of residues with amino acid "curr_amino"
        amino_one = zeros(num_seq,1);
        amino_one(loc_amino)=1;
        
        weight_one = amino_one.*weight_seq;
        freq_single(ind_amino) = sum(weight_one)/num_patients;
    end
    [freq_sort indsort]  = sort(freq_single,'descend');
    freq_single_array{ind_res} =freq_sort;
    amino_single_array{ind_res} = amino_all_unique(indsort);
end