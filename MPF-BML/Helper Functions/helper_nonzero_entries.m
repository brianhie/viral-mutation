function [xstartpos darraystartNonzeroPos darrayNonzeroPos darrayNonzero dactvalues xarray xpos num_mutant_xarray num_mutant_dplusxarray] = helper_nonzero_entries(num_mutants_combine_array, msa_bin)
%HELPER_NONZERO_ENTRIES returns location of the non-zero entries in MSA_BIN (x*) and the
% non-zero entries in the one-bit flipped sequences in MSA_BIN (d*)

cumul_num_mutants_combine_array = cumsum(num_mutants_combine_array);
protein_length_aa = length(num_mutants_combine_array);

total_length = size(msa_bin,2); % number of sites in expanded binary matrix
num_sequences_unique = size(msa_bin,1); % number of sequences in expanded binary matrix
num_mutant_xarray = sum(msa_bin');

% Obtain locations of ones in msa_bin
nozerosxarray = length(nonzeros(msa_bin(:)));
xarray = zeros(1,nozerosxarray);
count=1;
xstartpos= zeros(1,num_sequences_unique);
xpos = zeros(1,num_sequences_unique);
tic
for aba=1:num_sequences_unique
    
    
    curr_vector = msa_bin(aba,:);
    ind = find(curr_vector==1);
    xarray(count:count+length(ind)-1)=ind; % location of ones in each sequence
    count = count + length(ind);
    xpos(aba)=length(ind); % number of ones in each sequence
    if (aba==1)
        xstartpos(aba)=1;
    else
        xstartpos(aba)=sum(xpos(1:aba-1))+1; % start position of each sequence in xarray
    end
    
end

% Obtain the location of ones in the sequences when each residue is flipped

for aba=1:50
    temp_matrix=[];
    temp_matrix = fliplr(eye(aba));
    temp_matrix = [zeros(1,size(temp_matrix,2)) ; temp_matrix ];
    bin_matrix{aba} =temp_matrix;
    
end

count2=1;
total_x_length=0;
total_d_length=0;

testbin=1;
total_dnumber=0;
clear d;
count3=1;
Xtotal=[];
num_mutant_dplusxarray = zeros(1,num_sequences_unique*size(msa_bin,2));
for aba=1:num_sequences_unique % go through each sequence
    curr_vector = msa_bin(aba,:);
    
    
    count=1;
    count2=1;
    
    for bcb=1:protein_length_aa % go through each site
        length_curr = num_mutants_combine_array(bcb);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
        % all flip only to dominant mutant
        % find the starting position of the bcbth site
        if (bcb>1)
            curr_start = cumul_num_mutants_combine_array(bcb-1)+1;
        else
            curr_start=1;
        end
        pos_data=curr_start:curr_start+length_curr-1;
        curr_data = curr_vector(curr_start:curr_start+length_curr-1);
        
        if testbin==1
            if (curr_data==0)
                dec_value=0;
            elseif (curr_data==1)
                dec_value=1;
            else
                dec_value = bi2de(fliplr(curr_data));
            end
            
        else
            dec_value = bi2de(fliplr(curr_data));
        end
        
        
        bin_value = dec_value;
        
        jj=1:length_curr;
        con_values = [0 2.^(jj-1)];
        indnow = find(bin_value==con_values);
        jjcurrvalues=[0 jj];
        jjcurrvalues(indnow)=[];
        con_values2=con_values;
        con_values(indnow) = [];
        
        if (testbin==1)
            curr_bin_matrix = bin_matrix{length_curr};
            add_bi_values = curr_bin_matrix(jjcurrvalues+1,:);
            
        else
            add_bi_values =   fliplr(de2bi(con_values,length_curr));
        end
        
        
        
        indone = find(curr_data==1);
        if (sum(curr_data)>0)
            add_bi_values(:,indone)=-1;
        end
        
        d(count:count+size(add_bi_values,1)-1,curr_start:curr_start+length_curr-1)=add_bi_values;
        count=count+size(add_bi_values,1);
        
    end
    
    
    d_store1{aba}=sparse(d);
    
    dxtemp = bsxfun(@plus,curr_vector,sparse(d));
    
    num_mutant_dplusxarray(count3:count3+size(d,1)-1) = sum(dxtemp');
    count3 = count3 + size(d,1);
    
end
Dtotal = sparse(cell2mat(d_store1'));

[indi indj] = find(Dtotal~=0);
[sorti indk] = sort(indi);
darrayNonzero = indj(indk);

darrayNonzeroPos = histc(indi,1:size(Dtotal,1));

Dtotal=Dtotal';
Darray= Dtotal(:);
dactvalues = nonzeros(Darray);

darraystartNonzeroPos = cumsum([1; darrayNonzeroPos]);
darraystartNonzeroPos=darraystartNonzeroPos(1:end-1);
