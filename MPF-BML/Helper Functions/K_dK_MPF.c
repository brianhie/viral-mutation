#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
#include "mex.h"
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
    
    double * dK;
    double * dK2;
    double * K;
    int count,aba;
    double expTerm;
    int num_sequences_unique, total_length, num_d2;
    double * xstartpos, * darraystartNonzeroPos , * darrayNonzeroPos, * darrayNonzero, * xarray;
    double * J_expand ,  * dactvalues;
    double Jtemp1, Jtemp2, dxTemp;
    int darrayTemp1, dNonzerotemp1, jvalue1, jvalue2, ivalue1, ivalue2;
    int xstartTemp, xposTemp;
    int * iarray;
    double * xpos;
    int bcb, indj;
    double jtempvalue;
    int indi;
    double exptemp;
    double * dvaluestemp;
    double tempTerm;
    double * prob_D;
    double * num_mutant_xarray;
    double * num_mutant_dplusxarray;
    int dist_length;
    double * dKlength;
    int countray;
//     double * k_expand;
    double energy_mutant;
    int index1, index2;
    // inputs
    num_sequences_unique = mxGetScalar(prhs[0]);
    total_length = mxGetScalar(prhs[1]);
    xstartpos = mxGetPr(prhs[2]);
    num_d2 = mxGetScalar(prhs[3]);
    darraystartNonzeroPos = mxGetPr(prhs[4]);
    darrayNonzeroPos = mxGetPr(prhs[5]);
    darrayNonzero= mxGetPr(prhs[6]);
    dactvalues=mxGetPr(prhs[7]);
    xarray=mxGetPr(prhs[8]);
    J_expand=mxGetPr(prhs[9]);
    xpos=mxGetPr(prhs[10]);
    prob_D=mxGetPr(prhs[11]);
        num_mutant_xarray=mxGetPr(prhs[12]);
        num_mutant_dplusxarray=mxGetPr(prhs[13]);
        dist_length= mxGetScalar(prhs[14]);
//         k_expand = mxGetPr(prhs[15]);
    //	 dxactvalues=mxGetPr(prhs[16]);
    /////////////////////////////
    plhs[0]= mxCreateDoubleMatrix(1,1,mxREAL);
    K =  mxGetPr(plhs[0]);
    
    plhs[1] = mxCreateDoubleMatrix(1,(mwSize) total_length*total_length,mxREAL);
    dK =  mxGetPr(plhs[1]);
    
//         plhs[2] = mxCreateDoubleMatrix(1,(mwSize) dist_length,mxREAL);
//     dKlength =  mxGetPr(plhs[2]);
    
//     plhs[2] = mxCreateDoubleMatrix(1,(mwSize) total_length*total_length,mxREAL);
//     dK2 =  mxGetPr(plhs[2]);
    /////////
    // iarray= malloc(sizeof(int)*length_dact+10);
    //dxvaluestemp= malloc(sizeof(double)*length_dact+10);
    //dvaluestemp= malloc(sizeof(double)*length_dact+10);
    ////////////////////////////////////////////////////
    
    count=0;
    countray=0;
    //dK = zeros(1,total_length*total_length);
    K[0]=0;
    for (aba=0;aba<num_sequences_unique;aba++) {
        
        xstartTemp=xstartpos[aba]-1;
        xposTemp = xpos[aba];
        for (bcb=0;bcb<num_d2;bcb++) {
            
            Jtemp1=0;
            Jtemp2=0;
            
            darrayTemp1 = darraystartNonzeroPos[count]-1;
            dNonzerotemp1=darrayNonzeroPos[count];
            
            for (indj=0;indj<dNonzerotemp1;indj++) {
                
                jvalue1 =darrayNonzero[darrayTemp1+indj];
                jtempvalue=dactvalues[darrayTemp1+indj];
                // calculate first term
                for (indi=0;indi<xposTemp;indi++) {
                    
                    ivalue1 = total_length*(xarray[xstartTemp+indi]-1)+1;
                    Jtemp1 = Jtemp1 + J_expand[ivalue1+jvalue1-2]*jtempvalue;
                    
                    
                }
                // calculate second term
                for (indi=0;indi<dNonzerotemp1;indi++) {
                    
                    ivalue2 = total_length*(darrayNonzero[darrayTemp1+indi]-1)+1;
                    // ivalue2 = total_length*(darrayNonzero[darrayTemp1+indi]-1)+1;
                    Jtemp2 = Jtemp2 + J_expand[ivalue2+jvalue1-2]*dactvalues[darrayTemp1+indi]*jtempvalue;
                }
                
            }
            //	printf("Jtemp1 is %f and Jtemp2 is %f\n",Jtemp1, Jtemp2);
            // tempTerm=(2*Jtemp1 + Jtemp2)/2;
            index1 = num_mutant_xarray[aba];
            index2 = num_mutant_dplusxarray[countray];
            energy_mutant = 0;
            countray=countray+1;
            exptemp = exp(-(2*Jtemp1 + Jtemp2-energy_mutant)/2);
            // printf("K is %f\n",exptemp);
            K[0] = K[0]+ prob_D[aba]*exptemp;
            
            //calculate dK
            for (indj=0;indj<dNonzerotemp1;indj++) {
                
                jvalue1 =darrayNonzero[darrayTemp1+indj];
                jtempvalue=dactvalues[darrayTemp1+indj];
                // calculate first term
                for (indi=0;indi<xposTemp;indi++) {
                    
                    ivalue1 = total_length*(xarray[xstartTemp+indi]-1)+1;
                    //Jtemp1 = Jtemp1 + J_expand[ivalue1+jvalue1-2]*jtempvalue;
                    dK[ivalue1+jvalue1-2] =  dK[ivalue1+jvalue1-2]-prob_D[aba]*jtempvalue*exptemp;
                    
                    
                }
                // calculate second term
                for (indi=0;indi<dNonzerotemp1;indi++) {
                    
                    ivalue2 = total_length*(darrayNonzero[darrayTemp1+indi]-1)+1;
                    // ivalue2 = total_length*(darrayNonzero[darrayTemp1+indi]-1)+1;
                    // Jtemp2 = Jtemp2 + J_expand[ivalue2+jvalue1-2]*dactvalues[darrayTemp1+indi]*jtempvalue;
                    dK[ivalue2+jvalue1-2] =  dK[ivalue2+jvalue1-2]-prob_D[aba]*dactvalues[darrayTemp1+indi]*jtempvalue*exptemp/2;
                    
                }
                
            }
//             dKlength[index1] = dKlength[index1]+prob_D[aba]*exptemp/2; 
//             dKlength[index2] = dKlength[index2]-prob_D[aba]*exptemp/2; 
//             // calculate dK
//             for (indi=0;indi<dxNonzerotemp1;indi++) {
//                 //printf("Dxarray is %d\n",dxarrayTemp1+indi);
//                 ivalue2 = dxarrayNonzero[dxarrayTemp1+indi];
//                 dxTemp=dxactvalues[dxarrayTemp1+indi];
//                 
//                 //  ivalue2=iarray[indi];
//                 //  dxTemp = dxvaluestemp[indi];
//                 // printf("ivalue2 is %d and dxTemp is %f\n",ivalue2,dxTemp);
//                 for (indj=0;indj<dNonzerotemp1;indj++) {
//                     jvalue2 =total_length*(darrayNonzero[darrayTemp1+indj]-1)+1;
//                     // tempTerm=dxTemp*dactvalues[darrayTemp1+indj]*exptemp;
//                     dK2[ivalue2+jvalue2-2] = dK2[ivalue2+jvalue2-2]-prob_D[aba]*dxTemp*dactvalues[darrayTemp1+indj]*exptemp;
//                     // dK[ivalue2+jvalue2-2] = dK[ivalue2+jvalue2-2]-dxTemp*dvaluestemp[indj]*exptemp;
//                 }
//             }
            
            
            count=count+1;
        }
        
        
        // dK[0]=3;
    }
    //ee(iarray);
    //mxFree(dxarrayNonzero);
    //mxFree(dxactvalues);
}




/*
 * mxFree(currSeq);
 * mxFree(newSeq);
 * mxFree(hCurr);
 * mxFree(JCurr);
 * mxFree(multMult);
 * mxFree(flipbitIndArray);
 * mxFree(seqMat);
 */

/*
 * mxFree(currSeq);
 * mxFree(newSeq);
 * mxFree(hCurr);
 * mxFree(JCurr);
 * mxFree(multMult);
 * mxFree(flipbitIndArray);
 * mxFree(seqMat);
 *
 */

// mexPrintf("here2");





//}

