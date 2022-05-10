#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to perform all of the analysis on the combined nuclear features
for the WSI cohort. The code below generates the supplementary plots for the top
nuclear features. Additionally, this file contains the training and AUC plot 
for the random forest classifier used in the supplement. The code is wrapped up
in functions so that it can be called by the FigureMasterScript file.

 Copyright (C) 2021, Rajaram Lab - UTSouthwestern 
    
    This file is part of cancres-2022-cancres-2022-intratumoral-heterogeneity-dl-paper.
    
    cancres-2022-cancres-2022-intratumoral-heterogeneity-dl-paper is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    
    cancres-2022-cancres-2022-intratumoral-heterogeneity-dl-paper is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    
    You should have received a copy of the GNU General Public License
    along with cancres-2022-cancres-2022-intratumoral-heterogeneity-dl-paper.  If not, see <http://www.gnu.org/licenses/>.
    
    Paul Acosta, 2022
"""
import os as os
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir

import sys
import numpy as np
from scipy.stats import mannwhitneyu
import pickle
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import yaml 

with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)

statAnnotDir=os.path.join(ROOT_DIR,projectPaths['StatAnnot'])
sys.path.insert(0, statAnnotDir)
from statannot import add_stat_annotation

from matplotlib.lines import Line2D

import yaml 
import random


import statsmodels.stats.multitest as statMt
import Util.plotUtils as PlUtils
#%% Define figure parameters

fontSize=figParams['fontsize']
mutColors=figParams['colors']
wtColors=figParams['wtColors']
mutCmp=figParams['mutCmaps']

#%% Define Data

def Define_Nuclear_Data_To_Use(gene, projectDir, projectPaths):
    
    # Define Training and Testing Folds
    allSampleFile = os.path.join(projectDir, projectPaths['Data']['AllSamples'] )
    allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)
    
    # Get the samples used in our validation cohort
    foldsIdx=os.path.join(projectDir, projectPaths['Data']['FoldsIdx'])
    with open(foldsIdx, 'rb') as f:
            folds = pickle.load(f)
    validIdx=folds[3]
    trainIdx=np.concatenate((folds[0],folds[1],folds[2]),axis=0)
    testSamples=allSamples.iloc[validIdx].svs.values
    trainSamples=allSamples.iloc[trainIdx].svs.values 
    
    # Load Saved Feature Files
    combinedFeaturesFile=os.path.join(projectPaths['DataDir'],
                                      projectPaths['Nuclear']['CombinedFeats'])
    [combinedFeatMat,svsList,allFeatNames]=pickle.load(open(combinedFeaturesFile,'rb'))
    numberOfFeatures=len(allFeatNames)
    
    # Get Gene loss status of files
    allSamples=allSamples.set_index('svs')
    isGeneWT=allSamples.loc[svsList][gene+'_Positive'].values 

    return testSamples,trainSamples,combinedFeatMat,svsList,allFeatNames,numberOfFeatures,isGeneWT


#%% Train Random Forest Classifier
def Random_Forest_Classifier(gene, projectDir, projectPaths):
    data=Define_Nuclear_Data_To_Use(gene, projectDir, projectPaths)
    testSamples,trainSamples,combinedFeatMat,svsList,allFeatNames,numberOfFeatures,isGeneWT=data

    trainIdx=[svsList.index(sample) for sample in trainSamples if sample in svsList]
    testIdx=[svsList.index(sample) for sample in testSamples if sample in svsList]
    
    trainData=combinedFeatMat[trainIdx,:]
    testData=combinedFeatMat[testIdx,:]
    trainLabels=isGeneWT[trainIdx]
    testLabels=isGeneWT[testIdx]
    np.random.seed(1234)
    clf=RandomForestClassifier(n_estimators=500,
                                class_weight='balanced_subsample',
                                n_jobs=-1,random_state=random.seed(1234))    
    
    clf.fit(trainData,trainLabels)
    
    testScoresPredicted=clf.predict_proba(testData)[:,1]
    
    PlUtils.ROC_Plot(testLabels,testScoresPredicted,mutColors[gene],gene)


# %% Calculate p-value difference between WT and Loss samples for each feature
    
def Calculate_Pval_Difference(gene, data):
    testSamples,trainSamples,combinedFeatMat,svsList,allFeatNames,numberOfFeatures,isGeneWT=data
    featPVals=np.zeros(numberOfFeatures)
    for featureNum in range(numberOfFeatures):
        feat=combinedFeatMat[:,featureNum]
        u_stat,pval=mannwhitneyu(feat[isGeneWT],feat[np.logical_not(isGeneWT)],alternative='two-sided')
        featPVals[featureNum]=pval
    featPVals=np.array(featPVals)
    featOrderPval=np.argsort(featPVals)
    return featPVals, featOrderPval

#%% Generate Nuclear Feature Supplementary figures - Extract correlation
    
def Plot_Important_Features(gene, projectDir, projectPaths): 
    topPvalFeats={}
    pValFeatureNames={}

    data=Define_Nuclear_Data_To_Use(gene, projectDir, projectPaths)
    testSamples,trainSamples,combinedFeatMat,svsList,allFeatNames,numberOfFeatures,isGeneWT=data
    
    featPVals, featOrderPval= Calculate_Pval_Difference(gene,data)
    
    featCorrs=np.corrcoef(np.transpose(combinedFeatMat))
    selectedFeats=[]
    corrThresh=0.95
    nFeatsToShow=9
    nFeats=0
    i=0
    
    r=[]
    
    while nFeats<nFeatsToShow:
        isRepresented=False
        for j in range(i):
            if abs(featCorrs[featOrderPval[i],featOrderPval[j]])>corrThresh:
                         isRepresented=True
        if not isRepresented:
            selectedFeats.append(featOrderPval[i])                 
            nFeats=nFeats+1
        i=i+1    
    selectedFeats=np.array(selectedFeats)    
    topPvalFeats[gene]=selectedFeats
    
    pVals=featPVals[selectedFeats]
    correctedPvals=statMt.multipletests(pvals=pVals,method='bonferroni')[1]
    
    # Define simplified class names for plotting purposes 
    simplifiedNamesClasses={'Avg_Area':['Nuclear_Area','Nuclear Size'],
                     'Avg_BBox_Area':['Bounding_Box_Area','Nuclear Size'],
                     'Avg_Equivalent_Diameter':['Equivalent_Diameter','Nuclear Size'],
                     'Avg_Eccentricity':['Eccentricity','Nuclear Shape'],
                     'Avg_Major_Axis_Length':['Major_Axis_Length','Nuclear Size/Shape'],
                     'Avg_Convex_Area':['Convex_Area','Nuclear Size/Shape'],
                     'Avg_Solidity':['Solidity','Nuclear Shape'],
                     'Avg_H_Int_Mean':['H_Int_Mean','Hematoxylin\nIntensity'],
                     'Avg_H_Int_Median':['H_Int_Median','Hematoxylin\nIntensity'],
                     'Avg_H_Int_Std':['H_Int_Std','Hematoxylin\nDistribution'],
                     'Avg_H_Int_MAD':['H_Int_MAD','Hematoxylin\nDistribution'],
                     'Avg_H_Int_Min':['H_Int_Min','Hematoxylin\nIntensity'],
                     'Avg_H_Int_Max':['H_Int_Max','Hematoxylin\nIntensity'],
                     'Avg_H_Int_Kurtosis':['H_Int_Kurtosis','Hematoxylin\nDistribution'],
                     'Avg_H_Int_Skewness':['H_Int_Skewness','Hematoxylin\nDistribution'],
                     'Avg_LabL_Int_Mean':['LabL_Int_Mean','Lightness\nIntensity'],
                     'Avg_LabL_Int_Median':['LabL_Int_Median','Lightness\nIntensity'],
                     'Avg_LabL_Int_Std':['LabL_Int_Std','Lightness\nDistribution'],
                     'Avg_LabL_Int_MAD':['LabL_Int_MAD','Lightness\nDistribution'],
                     'Avg_LabL_Int_Min':['LabL_Int_Min','Lightness\nIntensity'],
                     'Avg_LabL_Int_Max':['LabL_Int_Max','Lightness\nIntensity'],
                     'Avg_LabL_Int_Kurtosis':['LabL_Int_Kurtosis','Lightness\nDistribution'],
                     'Avg_LabL_Int_Skewness':['LabL_Int_Skewness','Lightness\nDistribution'],
                     'Avg_Hint_Haralick_2nd_moment':['Haralick_2nd_moment','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_contrast':['Haralick_contrast','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_correlation':['Haralick_correlation','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_variance':['Haralick_variance','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_inv_diff_moment':['Haralick_inv_diff_moment','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_sum_avg':['Haralick_sum_avg','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_sum_variance':['Haralick_sum_variance','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_sum_entropy':['Haralick_sum_entropy','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_entropy':['Haralick_entropy','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_diff_var':['Haralick_diff_var','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_diff_entropy':['Haralick_diff_entropy','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_inf_corr1':['Haralick_inf_corr1','Hematoxylin\nTexture'],
                     'Avg_Hint_Haralick_inf_corr2':['Haralick_inf_corr2','Hematoxylin\nTexture'],}
    
    # Plot the top nuclear features for the indicated gene 
    w=0.4
    plt.figure(figsize=(20,15))
    for featCounter,featNum in enumerate(topPvalFeats[gene]):
        ax=plt.subplot(3,3,featCounter+1)
        
        status=[]
        
        for j in range(isGeneWT.size):
            if isGeneWT[j]==0:
                stat='Loss'
            else: stat='WT'
            status.append(stat)
        
        data=pd.DataFrame({gene:status,
                           'Feat':combinedFeatMat[:,featNum]})
        
       
        h=sns.violinplot(data=data,x=gene,y='Feat',
                       hue=gene,palette=[wtColors['BAP1'],mutColors[gene]],order=['Loss','WT'],
                       dodge=True,width=w*1.5,scale='width')
        
        
        ax,testResults=add_stat_annotation(h,data=data,x=gene,y='Feat',
                                           box_pairs=[('Loss','WT')],  line_offset_to_box=0.01,
                                           pvalues=[correctedPvals[featCounter]],perform_stat_test=False,
                                           text_format='full',verbose=1,
                                           loc='outside',fontsize=14,comparisons_correction=None)    
        r.append(testResults[0].pval)
        h.legend_.remove()
        plt.xticks([0.15,1-0.15],['Loss','WT'],fontsize=fontSize)
       
    
       
        plt.ylabel(simplifiedNamesClasses[allFeatNames[featNum]][0] , fontsize=16)
        legend_elements = [Line2D([0], [0], marker='o', color='w', 
                                  label=simplifiedNamesClasses[allFeatNames[featNum]][1],
                          markerfacecolor='w', markersize=1)]
        ax.annotate(simplifiedNamesClasses[allFeatNames[featNum]][1],
                    xy=(260, 260), xycoords='axes points',
            size=16, ha='right', va='top',
            bbox=dict(boxstyle='round', fc='w'))
    
        plt.xlabel('Status',fontsize=fontSize)
        plt.tight_layout()
    pValFeatureNames[gene]=np.array(allFeatNames)[topPvalFeats[gene]]


    
