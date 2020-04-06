function pdf_MSA = helper_PN(prob_D, msa_aa_ex)

total_mass = sum(prob_D);
mutant_MSA_prelim = sum(msa_aa_ex');

max_mutant = max(mutant_MSA_prelim);
for aba = 1:max_mutant
    ind_find = find(mutant_MSA_prelim == aba);
    partial_mass = sum(prob_D(ind_find));
    pdf_MSA(aba) = partial_mass / total_mass;
end