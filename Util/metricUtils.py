"""
metricUtils contains functions that allow you to calculate metrics reported 
throughout the paper. These metrics include accuracy, npv, precision and others.
It also has the function to calculate the correlation between gene classifications.

 Copyright (C) 2021, Rajaram Lab - UTSouthwestern 
    
    This file is part of cancres-2022-intratumoral-heterogeneity-dl-paper.
    
    cancres-2022-intratumoral-heterogeneity-dl-paper is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    cancres-2022-intratumoral-heterogeneity-dl-paper is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with cancres-2022-intratumoral-heterogeneity-dl-paper.  If not, see <http://www.gnu.org/licenses/>.
    
    Paul Acosta, Vipul Jarmale 2022
"""
import os, sys
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir
sys.path.insert(0, ROOT_DIR)

import numpy as np

from sklearn.metrics import average_precision_score,precision_score
from sklearn.metrics import confusion_matrix,f1_score, recall_score,matthews_corrcoef
from sklearn.metrics import hamming_loss,jaccard_score,balanced_accuracy_score,accuracy_score

import scipy

from scipy.stats import ks_2samp
from scipy.stats import rankdata
import pandas as pd 
import yaml


with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)

import Util.plotUtils as PlUtils


#%%

def CalculateMetrics(groundTruth,activations,threshold=0.5):
    """
    This function is used to calculate several of the metrics reported in the paper.
    Not all metrics calculated here are used in the final paper. Please note that you 
    will need to load the data through cohort specific functions found in plotUtils.py
    before being able to use this function. 

    Parameters
    ----------
    groundTruth : np.array
        Ground truth values for the samples.
    activations : np.array
        Activation scores for the samples.
    threshold : float16, optional
        Cutoff threshold for stratification of the continuous actvations. 
        The default is 0.5.

    Returns
    -------
    CM : np.array
        Confusion matrix.
    mat_met : pd.DataFrame
        Dataframe containing all of the metrics calculated for the data provided

    """

    # Filter nan values for calculations
    filteredGroundTruth=groundTruth[~np.isnan(activations)]
    filteredPredictions=activations[~np.isnan(activations)]   
    
    # Define threshold of classification 
    filteredPredictions[filteredPredictions<threshold]=0
    filteredPredictions[filteredPredictions>=threshold]=1
    
    # invert all values to match clinical meanming of TP and TN
    invertGroundTruth=1-filteredGroundTruth
    invertPredictions=1-filteredPredictions
        
    # Calculate metrics
    CM = confusion_matrix(invertGroundTruth, invertPredictions)
    PPV = precision_score(invertGroundTruth,invertPredictions)
    
    tn, fp, fn, tp = CM.ravel()
    NPV = tn/(tn+fn)
    Specificity = tn/(tn+fp)
 
    
    Precision = precision_score(invertGroundTruth, invertPredictions)
    F1 = f1_score(invertGroundTruth,invertPredictions)
    Accuracy = accuracy_score(invertGroundTruth,invertPredictions)     
    Recall = recall_score(invertGroundTruth,invertPredictions)   # same as sensitivity
    MCC = matthews_corrcoef(invertGroundTruth,invertPredictions)  
    Hamming = hamming_loss(invertGroundTruth,invertPredictions)
    Jaccard = jaccard_score(invertGroundTruth,invertPredictions) 
    Prec_Avg = average_precision_score(invertGroundTruth,invertPredictions) 
    Accu_Avg = balanced_accuracy_score(invertGroundTruth,invertPredictions) 
    
    mat_met = pd.DataFrame({'Metric': ['Accuracy','PPV','NPV','Specificity',
                                       'Precision','Recall','F1','MCC','Hamming',
                                       'Jaccard','Precision_Avg','Accuracy_Avg'],
                            'Value': [Accuracy,PPV,NPV,Specificity,Precision,
                                      Recall,F1,MCC,Hamming,Jaccard,Prec_Avg,
                                      Accu_Avg]})
    
    return CM,mat_met

def GeneIndependenceTests(genePair):
    """
    This function is used to calculate the correlation between two gene predictions.
    It also calculates the KS value, but that value is not used in the final paper.

    Parameters
    ----------
    genePair : list
        List containing two gene names. i.e. ['BAP1','PBRM1']

    Returns
    -------
    similarCurves : 
        Result of KS test. Not used
    differentCurves : 
        Result of KS test. Not used.
    correlation : 
        Spearman correlation and pvals.

    """
    gene1,gene2=genePair
    cohort='WSI'
   
    geneActivations,geneIsWT,geneIsFocal={},{},{}
    for gene in figParams['geneList']:
        predScore,isWt,isFocal,_,_=PlUtils.GetSlideLevelResults(cohort, gene)
        geneActivations[gene]=predScore
        geneIsFocal[gene]=isFocal
        geneIsWT[gene]=isWt
    
    # Only non-focal (i.e. not LL) UL or WT
    isGood=np.logical_and(np.logical_not(geneIsFocal[gene1]),
                          np.logical_not(geneIsFocal[gene2]))
    # Only samples with not NAN activation
    isGood=np.logical_and(isGood,np.logical_not(np.logical_or(\
                        np.isnan(geneActivations[gene1]),\
                            np.isnan(geneActivations[gene2]))))
        
    geneState=(1-np.int16(geneIsWT[gene1]))+2*(1-np.int16(geneIsWT[gene2]))
    X=rankdata(1-geneActivations[gene1][isGood])/np.sum(isGood)
    Y=rankdata(1-geneActivations[gene2][isGood])/np.sum(isGood)
    hueClass=geneState[isGood]
    # hue = 0 : both genes are WT
    # hue = 1 : gene1 loss, gene2 WT
    # hue = 2 : gene1 WT, gene 2 Loss
    # hue = 3 : gene1 Loss, gene 2 Loss
    similarCurves=ks_2samp(X[hueClass==0],
                           np.array(list(X[hueClass==2])+list(X[hueClass==3])))
    
    differentCurves=ks_2samp(X[hueClass==0],X[hueClass==1])
    
    correlation=scipy.stats.spearmanr(X,Y)
    # Please note that although we calculate the ks value for the curves, this is 
    # not used in the paper
    return similarCurves,differentCurves,correlation    

