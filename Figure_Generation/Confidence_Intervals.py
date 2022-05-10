#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to generate confidence intervals for our AUC evaluation metrics 
for all gene models. Here, we implement a bootstrapping method with 10,000 bootstraps
to calculate a 95% confidence interval.

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
    
    Paul Acosta, 2022
"""
import os 
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir

import numpy as np
import pandas as pd
import yaml
import os as os

with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)
import Util.plotUtils as PlUtils

# Define plotting parameters 
fontSize=figParams['fontsize']
mutColors=figParams['colors']
wtColors=figParams['wtColors']
geneList=figParams['geneList']

#%%


def Calculate_CI(classifier,cohort,gene,nBootStraps=10000,bootStrapProportion=1.0):
    if classifier=='slide':
        geneActivations,geneIsWT,_,_,_=PlUtils.GetSlideLevelResults(cohort, gene)
 
    else:
        if cohort=='WSI':
            tessellationStride=figParams['RegionParams']['tessellationStride']
            minTumorThreshold=figParams['RegionParams']['threshold']
        
   
            regional=PlUtils.RegionalResponse(gene,tessellationStride,minTumorThreshold)
            geneIsWT,geneActivations, _=regional.Get_WSI_Response()
            
        else:
            assert(gene=='BAP1')
            responseInfo=PlUtils.Get_TMA_Response(cohort=cohort,isNormalized=True)
            if cohort=='TMA2':
                responseInfo=PlUtils.TMA_Patient_Level_Response(responseInfo)
            
            geneActivations=responseInfo.BAP1_Response.values
            geneIsWT=responseInfo.isBAP1WT.values
            
    nSamples=len(geneActivations)
    aucList=[]
    for n in range(nBootStraps):
        subset=np.random.choice(np.arange(nSamples),
                                size=np.int(np.round(nSamples*bootStrapProportion)),
                                replace=True)
        activations=geneActivations[subset]
        isWT=geneIsWT[subset]
        auc,_,_=PlUtils.ROC_Plot(np.uint8(isWT),activations, mutColors[gene],gene,plot=False) 
        aucList.append(auc)
    
    CI=[np.percentile(aucList,[2.5]),np.percentile(aucList,[97.5])]    
    
    avgAUC=np.mean(aucList)
    return CI,avgAUC
# %%
dataSets={'slide':{'WSI':['BAP1','PBRM1','SETD2'],
                   'TCGA':['BAP1','PBRM1','SETD2']},
          'region':{'WSI':['BAP1','PBRM1','SETD2'],
                     'TMA1':['BAP1'],'TMA2':['BAP1'],
                     'PDX1':['BAP1']}}                                               
classifierList=[]
cohortList=[]
geneList=[]
aucList=[]
ciList=[]
for classifier in dataSets:
    for cohort in dataSets[classifier]:
        for gene in dataSets[classifier][cohort]:
            CI,avgAUC=Calculate_CI(classifier,cohort,gene)
            print(classifier,cohort,gene,avgAUC,CI)
            classifierList.append(classifier)
            cohortList.append(cohort)
            geneList.append(gene)
            aucList.append(avgAUC)
            ciList.append(CI)
            
aucTable=pd.DataFrame({'Model':classifierList,
                       'Cohort':cohortList,
                       'Gene':geneList,
                       'AUC':aucList,
                       'CI(95%)':ciList})            
