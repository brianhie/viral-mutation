#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "mex.h"
#include <time.h>
//#include "rvgs.c"
//#include "rngs.c"
#include "matrix.h"
//#include "wnrnd.h"
#define NEW(x) (x*)mxMalloc(sizeof(x))
//#define NEW2(x) (x**)mxMalloc(sizeof(x*))
//#define maxSamples 100000
//#define aminoLength 7

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[]) {
    
//     long long int aba, nosim;
    int count, indi, indj,  rand_site, rand_amino, curr_start, curr_end, curr_change_pos, flag, onePosChange;
    int burnin, thin, total_length;
    unsigned long nosim, aba;
    double energy_curr, energy_new, trans_value, trans_prob;
    int * dOnePos;
    int * dvalue;
    
    double * double_sum;
    double * curr_vector;
    double * J_MINFLOW_mat_array;
    double currvalue;
    int * onepos;
    int onepos_length;
    int ivalue_onepos;
    int jvalue_onepos;
    double * phi_cumulative;
    double * phi_curr;
    double * num_samples;
    double * samples_output;
    int sample_length;
    int protein_length;
    int phicurrtemp;
    double random_part;
    double * num_mutants;
    int dist_length;
//     double * k_expand;
//     double * partition;
    double binary_value;
    double * binary_value_array;
    int seedtime;
//     int multTerm;
// inputs
    
    curr_vector = mxGetPr(prhs[0]);
    J_MINFLOW_mat_array = mxGetPr(prhs[1]);
    total_length = mxGetScalar(prhs[2]);
    nosim = (unsigned long) mxGetScalar(prhs[3]);
    phi_cumulative= mxGetPr(prhs[4]);
    phi_curr= mxGetPr(prhs[5]);
    burnin= mxGetScalar(prhs[6]);
    thin= mxGetScalar(prhs[7]);
//     curr_vector= mxGetPr(prhs[8]);
    sample_length= mxGetScalar(prhs[8]);
    protein_length = mxGetScalar(prhs[9]);
//     k_expand = mxGetPr(prhs[11]);
//         multTerm = mxGetScalar(prhs[11]);
/////////////////////////////
    plhs[0]= mxCreateDoubleMatrix(1,(mwSize) total_length*total_length,mxREAL);
    double_sum =  mxGetPr(plhs[0]);
    
    plhs[1]= mxCreateDoubleMatrix(1,1,mxREAL);
    num_samples =  mxGetPr(plhs[1]);
    
    plhs[2]= mxCreateDoubleMatrix(1,(mwSize) sample_length,mxREAL);
    num_mutants =  mxGetPr(plhs[2]);
    
// plhs[3]= mxCreateDoubleMatrix(1,(mwSize) (sample_length),mxREAL);
// partition =  mxGetPr(plhs[3]);
// 
// plhs[4]= mxCreateDoubleMatrix(1,(mwSize) (sample_length),mxREAL);
// binary_value_array =  mxGetPr(plhs[4]);
    
////////////////////////////////////////////////////
    onepos=mxMalloc(sizeof(int)*total_length+total_length);
    dOnePos=mxMalloc(sizeof(int)*3);
    dvalue=mxMalloc(sizeof(int)*3);
//////////////////
    count=0;
    for (aba=0;aba<total_length;aba++) {
        currvalue = curr_vector[aba];
        //printf("Currvalue is %f and aba is %d\n",currvalue,aba);
        if (currvalue==1){
            onepos[count]=aba;
            //printf("Onepos is %d\n",onepos[count]);
            count=count+1;
            
        }
    }
    onepos_length=count;
//printf("onepos_length is %d\n",onepos_length);
    
   // binary_value=0;
    energy_curr=0;
    for (indi=0;indi<onepos_length;indi++) {
        ivalue_onepos = onepos[indi];
        for (indj=0;indj<onepos_length;indj++) {
            jvalue_onepos= onepos[indj];
            energy_curr = energy_curr + J_MINFLOW_mat_array[total_length*ivalue_onepos+jvalue_onepos];
        }
          //  binary_value = binary_value + 2^ivalue_onepos;

    }
    
//     energy_curr = energy_curr + k_expand[onepos_length];
    
    dOnePos[0]=1;
    dOnePos[1]=1;
    dvalue[0]=0;
    dvalue[0]=1;
    
    count=0;
    
  //  printf("nosim is %d",nosim);
  srand(time(NULL));
//     seedtime=2;
//       srand((unsigned) seedtime);


    for (aba=0;aba<nosim;aba++) {
        
        //  rand_site = 0;
        rand_site=  rand() % protein_length;
        //rand_site = random_site_array[aba]-1; // random site starting from zero
//	rand_amino=rand_amino_array[aba]-1; // random amino in the site
        phicurrtemp = phi_curr[rand_site];
        rand_amino = rand() % phicurrtemp;
//printf("rand is %d\n",rand_site);
        
        
//length_curr = phi_curr[rand_site];
        if (rand_site>0) // find the start of the random site
            curr_start = phi_cumulative[rand_site-1];
        else
            curr_start=0;
        
        
        
        curr_end = curr_start + phi_curr[rand_site]-1;
        curr_change_pos = curr_start + rand_amino;
        
        
        //  printf("curr start is %d\n",curr_start);
        
        
        //printf("start is %d and end is %d and curr_change_pos is %d\n",curr_start,curr_end,curr_change_pos);
        /////////////////////////////
        // form d value
        
        
        
        // cancel d part
        flag=0;
        dOnePos[0]=-99;
        onePosChange=-99;
        for (indi=0;indi<onepos_length;indi++) {
            if ((onepos[indi]<=curr_end) && (onepos[indi]>=curr_start)) // if  the current vector is NOT the wildtype in the amino acid locations
            {
                dOnePos[0]= onepos[indi]; // if there is a one, store the location of the one
                dvalue[0]= -1; // have to subtract the one to get new vector
                flag=1;
                onePosChange=indi; // the position in onepos which is a one and needs changing
            }
        }
        if (flag==0) // if  the current vector is the wildtype in the amino acid locations
            dvalue[0]=0; // don't have to substract anything
        
        // add d part
        dOnePos[1] = curr_change_pos; // location where the bit is going to be flipped
        if(dOnePos[1]!=dOnePos[0])
            dvalue[1] = 1; // if flip from zero to one
        else
            dvalue[1] = 0; //        if flip from one to zero
        
        //printf("energy curr is %f\n",energy_curr);
        //printf("aba is %d\n,",aba);
//	printf("donepos is %d %d and dvalue is %d %d and oneposchange is %d and oneposlength is %d\n",dOnePos[0],dOnePos[1],dvalue[0],dvalue[1],onePosChange,onepos_length);
        
        //for (indj=0;indj<onepos_length;indj++) {
        //printf("onepos %d is %d \n",indj,onepos[indj]);
        //}
        
        energy_new=energy_curr;
        if(  dOnePos[0]==-99)
        {
            
            indi=1;
            indj=1;
            
            energy_new = energy_new + J_MINFLOW_mat_array[total_length*dOnePos[indi]+dOnePos[indj]]*dvalue[indi]*dvalue[indj];
            
            for (indj=0;indj<onepos_length;indj++) {
                energy_new = energy_new + 2*J_MINFLOW_mat_array[total_length*dOnePos[indi]+onepos[indj]]*dvalue[indi];
            }
            
        }
        else
        {
            
            //printf("here=1");
            for (indi=0;indi<2;indi++) {
                //	printf("energy_new is %f ",energy_new);
                for (indj=0;indj<2;indj++) {
                    energy_new = energy_new + J_MINFLOW_mat_array[(total_length*dOnePos[indi])+dOnePos[indj]]*dvalue[indi]*dvalue[indj];
                }
                for (indj=0;indj<onepos_length;indj++) {
                    energy_new = energy_new + 2*(J_MINFLOW_mat_array[(total_length*dOnePos[indi])+onepos[indj]]*dvalue[indi]);
                }
                
            }
        }
        
        ////
        
//         
//         if (dOnePos[1]!=dOnePos[0]) // if not change to the wildtype
//             
//         {
//             
// //             if (dvalue[0]!=0) // if the current vector is not the wildtype
//                 
//               //  energy_new = energy_new -  k_expand[onepos_length];
//             if (dvalue[0]==0) { // if the current vector is the wildtype
//                 
//                 //onepos[onepos_length]=curr_change_pos; // add entry to end
// //        			onepos_length=onepos_length+1;
//                 energy_new = energy_new +  k_expand[onepos_length+1]-  k_expand[onepos_length];
                
//             }
//             
//         }
//         
//         
//         else { // if change to the wildtype
//             // shift entry back one place
//             //for (indi=onePosChange;indi<onepos_length-1;indi++) {
//             //		onepos[indi] = onepos[indi+1];
//             //}
//             //onepos[onepos_length]=-99;
//             //onepos_length=onepos_length-1;
//             energy_new = energy_new +  k_expand[onepos_length-1]-  k_expand[onepos_length];
//             
//             
//         }
//         
        
        /////
        
        trans_value = 1/(1+exp((-energy_curr+energy_new)));
        
        if (1<trans_value)
            trans_prob=1;
        else
            trans_prob = trans_value;
        
        //printf("energy_curr is %f, energy_new is %f and transprob is %f\n\n",energy_curr,energy_new,trans_prob);
        //printf("energy_new is %f\n",energy_new);
        
//	printf("aba is %d and thin is %d and burnin is %d\n",aba,thin,burnin);
        
        // thinparam = (aba+1) % thin;
        //	printf("aba is %d and thin is %d\n",aba,thin);
        
        //  if (aba<burnin)
        //  printf("aba is %d",aba);
        
        if (aba>=burnin)
        {
            //printf("here");
            if ( (aba+1) % thin == 0)
            {
                //	printf("aba is %d and thin is %d\n",aba,thin);
                
                //             samples2(count,:) =curr_vector_bin;
//                 binary_value=0;
                for (indi=0;indi<onepos_length;indi++) {
                    
                        ivalue_onepos = onepos[indi];
                    
                    double_sum[total_length*onepos[indi]+onepos[indi]] = double_sum[total_length*onepos[indi]+onepos[indi]]+1;
                    
                    for (indj=indi+1;indj<onepos_length;indj++) {
                        double_sum[total_length*onepos[indi]+onepos[indj]] = double_sum[total_length*onepos[indi]+onepos[indj]]+1;
                    }
                    	//samples_output[count*total_length+onepos[indi]]=1;
//                      binary_value = binary_value +  (int) pow((double) 2,ivalue_onepos);
                }
//                 binary_value_array[count] = binary_value;
                num_mutants[count] = onepos_length;
//                     partition[count] = energy_curr;
                count=count+1;
            }
        }
     
        //prinf("here");
        random_part = rand() / (double) RAND_MAX;
        if (random_part<trans_prob) // change
        {
            //curr_vector_bin[dOnePos(0)] = curr_vector_bin[dOnePos[0]]+dvalue[0];
            //curr_vector_bin[dOnePos[1]] = curr_vector_bin[dOnePos[1]]+dvalue[1];
            
            
            // change OnePos
            if (dOnePos[1]!=dOnePos[0]) // if not change to the wildtype
                
            {
                
                if (dvalue[0]!=0) // if the current vector is not the wildtype
                    
                    onepos[onePosChange]=curr_change_pos; // replace entry in onePos with new entry
                else { // if the current vector is the wildtype
                    
                    onepos[onepos_length]=curr_change_pos; // add entry to end
                    onepos_length=onepos_length+1;
                    
                }
                
            }
            
            
            else { // if change to the wildtype
                // shift entry back one place
                for (indi=onePosChange;indi<onepos_length-1;indi++) {
                    onepos[indi] = onepos[indi+1];
                }
                onepos[onepos_length]=-99;
                onepos_length=onepos_length-1;
                
            }
            energy_curr=energy_new;
            
        }      /*
        */
        
        
    }
    
         //    for (indi=0;indi<onepos_length;indi++) {
         //     samples_output[count*total_length+onepos[indi]]=1;

          //   }

//printf("Onepos_length is %d and size is %f",onepos_length,sizeof(onepos));
    
//mxFree(onepos);
//free(dOnePos);
//free(dvalue);
    num_samples[0] = count;
    //  double_sum[0]=1;
  
}





