#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Average Tessellation is used to combine the outputs of all three gene fold models for each cohort. 
i.e. The outputs of BAP1_0F_Region, BAP1_1F_Region, and BAP1_2F_Region on WSI are combined 
to create a single output for each sample.
    

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
sys.path.insert(0,ROOT_DIR)

import pickle
import sys
import numpy as np

import pandas as pd 
import yaml

sys.path.insert(0, os.path.join(ROOT_DIR,'External/tiler/'))

with open(os.path.join(ROOT_DIR, 'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Data_Files/TMA/Cohort_Files.yaml')) as file:
    tmaFiles = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Model_Files.yaml')) as file:
    modelFiles = yaml.full_load(file)

dataDir=projectPaths['DataDir']
modelDir=os.path.join(dataDir,modelFiles['FCN_Models']['BaseDir'])
responseData=os.path.join(dataDir,projectPaths['ResponseData'])




#%%
def Calculate_Average(cohort,gene='BAP1',norm=True):
    
    if cohort!='WSI':
       
        if norm:
            saveDir=os.path.join(responseData,'Region_Level','TMA_Norm/')
        else:
            saveDir=os.path.join(responseData,'Region_Level','TMA_NoNorm/')    
            
        baseSvsDir=os.path.join(dataDir,
                                projectPaths['Data']['ImageData'],
                                'TMA/')
        filesToUse=tmaFiles[cohort]
        svsList=[os.path.join(baseSvsDir,svs) for svs in filesToUse]
        
    else:
        baseSvsDir=os.path.join(dataDir,projectPaths['Data']['ImageData'],'WSI/')
        
        saveDir=os.path.join(responseData, 'Region_Level', 'WSI/', gene)

        
        allSampleFile=os.path.join(ROOT_DIR,projectPaths['Data']['AllSamples'])
        foldsIdx=os.path.join(ROOT_DIR,projectPaths['Data']['FoldsIdx'])
        allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)
        with open(foldsIdx, 'rb') as f:
            folds = pickle.load(f)
        validIdx=folds[3]
        testSamples=allSamples.iloc[validIdx]
        
        # Create svs list 
        svsList=[os.path.join(baseSvsDir,svs) for svs in testSamples.svs.values]
    
    #Loop over files to average activations
    for svsCounter,svs in enumerate(svsList):
        avgPklFile=os.path.join(saveDir, os.path.split(svs)[-1].replace('.svs','_avg.pkl'))
        if os.path.exists(avgPklFile):
            print(avgPklFile, ' \nAlready Exists!\n')
            continue
        
        activations=[]        
        for foldNum in range(3):
            pklFile=os.path.join(saveDir, os.path.split(svs)[-1].replace('.svs','_Fold-'+str(foldNum)+'.pkl'))
            slideActivations=pickle.load(open(pklFile,'rb'))
            activations.append(slideActivations)
    
        avgActivation=np.mean(np.stack(activations),axis=0)
        avgClasses=np.argmax(avgActivation,axis=-1)
        
        with open(avgPklFile, 'wb') as fh:
            pickle.dump([avgClasses, avgActivation], fh)

            
#%%

# Average all cohorts used in the main paper  (TMA cohorts with normalization)
for cohort in ['WSI','TMA1','TMA2','PDX1']:
    if cohort ==  'WSI':
        geneList=['BAP1','PBRM1','SETD2']
    else:
        geneList=['BAP1']
        
        
    for gene in geneList:
        Calculate_Average(cohort,gene)
        
# Average all TMA cohorts without normalization for the supplementary figures
for noNormCohort in ['TMA1','TMA2','PDX1']:
    Calculate_Average(noNormCohort,gene='BAP1', norm=False)
