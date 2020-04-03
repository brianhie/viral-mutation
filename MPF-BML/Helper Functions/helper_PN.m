function pdf_MSA = helper_PN(prob_D,msa_aa_ex)

num_entry = round(prob_D/min(prob_D));
mutant_MSA_prelim= sum(msa_aa_ex');
mutant_MSA=zeros(1,sum(num_entry));
count=1;
for aba=1:length(prob_D)
    mutant_MSA(count:count+num_entry(aba)-1)= ones(1,num_entry(aba))*mutant_MSA_prelim(aba);
    count=count+num_entry(aba);
end

max_mutant = max(mutant_MSA_prelim);
for aba=1:max_mutant
    ind_find=find(mutant_MSA==aba);
    pdf_MSA(aba) = length(ind_find)/length(mutant_MSA);
    
end