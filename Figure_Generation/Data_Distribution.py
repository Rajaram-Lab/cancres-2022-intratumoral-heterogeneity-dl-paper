#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to generate the data that populates supplementary tables 1-2.

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

import os as os
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir

import numpy as np
import pickle

import pandas as pd 
import yaml
import os as os


import progressbar

import Util.PatchGen as pg

with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR, 'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
    
import Util.plotUtils as PlUtils
fontSize=figParams['fontsize']
mutColors=figParams['colors']
wtColors=figParams['wtColors']
geneList=figParams['geneList']

#%% WSI data for WSI
dataToView='all'
geneData={}
isUsedIndex=[]
save=True 

allSamples=pd.read_csv(os.path.join(ROOT_DIR, projectPaths['Data']['AllSamples']))
folds=pickle.load(open(os.path.join(ROOT_DIR, projectPaths['Data']['FoldsIdx']),'rb'))

if dataToView=='training':
    data=np.concatenate((folds[:3]))
elif dataToView=='holdout':
    data=folds[3]
else:
    data=np.concatenate((folds[:]))

sampleData=allSamples.iloc[data]
# h5Dir='/project/bioinformatics/Rajaram_lab/shared/ccRCC/Patch_Data/BAP1/5X+20X/'
h5Dir=os.path.join(projectPaths['DataDir'],projectPaths['Data'],['PatchData'],
                   'WSI/')
samples=[os.path.join(h5Dir,svs.replace('svs','hdf5')) for svs in sampleData.svs.values]

sampleUsed=0

bar=progressbar.ProgressBar(max_value=len(samples))
for count,sample in enumerate(samples):
    patches=pg.LoadHdf5Data(sample)['patchData']
    if len(patches)!=0:
        sampleUsed+=1
        isUsedIndex.append(count)
    bar.update(count)
bar.finish() 

for gene in figParams['geneList']: 
    usedIsWt=np.sum(sampleData.iloc[isUsedIndex][gene+'_Positive'].values)
    usedIsFocal=np.sum(sampleData.iloc[isUsedIndex][gene+'_Focal'].values)
    
    usedIsLoss=sampleUsed-usedIsWt
    geneData[gene]={'Total':sampleUsed,'WT': usedIsWt,'Loss': usedIsLoss,'Focal':usedIsFocal}


# TCGA Data for Supp Table 1
geneDataTCGA={}
cohort='TCGA'
for gene in ['BAP1','PBRM1','SETD2']:
    predScore,isWt,isFocal,sampleList,scoresByFile=PlUtils.GetSlideLevelResults(cohort, gene)
    usedInAnalysis=np.logical_and(np.isfinite(predScore),np.isfinite(isWt))
    
    totalSamples=np.sum(usedInAnalysis)
    numWt=np.sum(isWt[usedInAnalysis]==1)
    numLoss=np.sum(isWt[usedInAnalysis]==0)
    
    geneDataTCGA[gene]={'Total': totalSamples,
                           'WT': numWt,'Loss': numLoss}

# Build table data
cohort=['WSI']*3+['TCGA']*3
geneList=figParams['geneList']*2

localizedLoss=[]
lossCases=[]
lossPcnt=[]
universalLoss=[]
llPcnt=[]
ulPcnt=[]
wtCases=[]
for count,gene in enumerate(geneList):
    if count in range(3):
        lossCases.append(geneData[gene]['Loss'])
        wtCases.append(geneData[gene]['WT'])
        lossPcnt.append(np.round(np.array(geneData[gene]['Loss']/geneData[gene]['Total'])*100,decimals=2))
        localizedLoss.append(geneData[gene]['Focal'])
        universalLoss.append(geneData[gene]['Loss']-geneData[gene]['Focal'])
        llPcnt.append(np.round(np.array(geneData[gene]['Focal']/geneData[gene]['Loss'])*100,decimals=2))
        ulPcnt.append(np.round(np.array((geneData[gene]['Loss']-geneData[gene]['Focal'])/geneData[gene]['Loss'])*100,decimals=2))
    else: 
        lossCases.append(geneDataTCGA[gene]['Loss'])
        wtCases.append(geneDataTCGA[gene]['WT'])
        lossPcnt.append(np.round(np.array(geneDataTCGA[gene]['Loss']/geneDataTCGA[gene]['Total'])*100,decimals=2))
        localizedLoss.append(np.nan)
        universalLoss.append(np.nan)
        llPcnt.append(np.nan)
        ulPcnt.append(np.nan)

table1Df=pd.DataFrame({'Cohort':cohort,
                       'Gene':geneList,
                       'WT Cases':wtCases,
                       'Loss Cases': lossCases,
                       'Loss Percentage': lossPcnt,
                       'UL Cases': universalLoss,
                       'UL Percentage': ulPcnt,
                       'LL Cases': localizedLoss,
                       "LL Percentage": llPcnt})
if save:
    table1Df.to_csv(os.path.join(ROOT_DIR, 'DataFiles/SlideDataDistribution.csv'))
#%% Supp Table 2

allSampleFile=os.path.join(ROOT_DIR, projectPaths['Data']['AllSamples'])
foldsIdx=os.path.join(ROOT_DIR, projectPaths['Data']['FoldsIdx'])
with open(foldsIdx, 'rb') as f:
    folds = pickle.load(f)

allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)

gene, mut, mutPnt, wt, wtPnt, focal, focalPnt=[],[],[],[],[],[],[]

for fold in range(len(folds)):
    # BAP1 
    fold1=allSamples.iloc[folds[fold]]
    
    samplesUsed=[]
    h5Dir=os.path.join(projectPaths['DataDir'],projectPaths['Data'],['PatchData'],
                   'WSI/')
    
    samples=[os.path.join(h5Dir,svs.replace('svs','hdf5')) for svs in fold1.svs.values]
    bar=progressbar.ProgressBar(max_value=len(samples))
    for count,sample in enumerate(samples):
        patches=pg.LoadHdf5Data(sample)['patchData']
        if len(patches)!=0:
            samplesUsed.append(True)
        else:
            samplesUsed.append(False)
        bar.update(count)
    bar.finish() 
    
    bap1WT=np.sum(fold1.BAP1_Positive.values[samplesUsed])
    bap1Total=fold1.BAP1_Positive.values[[samplesUsed]].size        
    bap1Focal=np.sum(fold1.BAP1_Focal.values[samplesUsed])
    bap1Focal_Pcnt=bap1Focal/bap1Total
    
    bap1Mut=np.sum(~fold1.BAP1_Positive.values[samplesUsed])-bap1Focal
    bap1Mut_Pcnt=bap1Mut/bap1Total
    bap1WT_Pcnt=bap1WT/bap1Total
    
    # PBRM1
    pbrm1WT=np.sum(fold1.PBRM1_Positive.values[samplesUsed])
    pbrm1Total=fold1.PBRM1_Positive.values[samplesUsed].size
    pbrm1Focal=np.sum(fold1.PBRM1_Focal.values[samplesUsed])
    pbrm1Focal_Pcnt=pbrm1Focal/pbrm1Total
    
    pbrm1Mut=np.sum(~fold1.PBRM1_Positive.values[samplesUsed])-pbrm1Focal
    pbrm1Mut_Pcnt=pbrm1Mut/pbrm1Total
    pbrm1WT_Pcnt=pbrm1WT/pbrm1Total
    
    #SETD2
    setd2WT=np.sum(fold1.SETD2_Positive.values[samplesUsed])
    setd2Total=fold1.SETD2_Positive.values[samplesUsed].size       
    setd2Focal=np.sum(fold1.SETD2_Focal.values[samplesUsed])
    setd2Focal_Pcnt=setd2Focal/setd2Total
    
    setd2Mut=np.sum(~fold1.SETD2_Positive.values[samplesUsed])-setd2Focal
    setd2Mut_Pcnt=setd2Mut/setd2Total
    setd2WT_Pcnt=setd2WT/setd2Total
    
    gene=gene+['BAP1','PBRM1','SETD2']
    mut=mut+list([bap1Mut,pbrm1Mut,setd2Mut])
    mutPnt=mutPnt+list([bap1Mut_Pcnt*100,pbrm1Mut_Pcnt*100,setd2Mut_Pcnt*100])
    wt=wt+list([bap1WT,pbrm1WT,setd2WT])
    wtPnt=wtPnt+list([bap1WT_Pcnt*100,pbrm1WT_Pcnt*100,setd2WT_Pcnt*100])
    focal=focal+list([bap1Focal,pbrm1Focal,setd2Focal])
    focalPnt=focalPnt+list([bap1Focal_Pcnt*100,pbrm1Focal_Pcnt*100,setd2Focal_Pcnt*100])
    
    
data_mat={'Gene':gene,'Mut Cases':mut,'Mut Percentage':mutPnt,'WT Cases':wt,
          'WT Percentage':wtPnt,'Focal Cases':focal,'Focal Percentage':focalPnt}
dataMat=pd.DataFrame(data_mat)

if save:
    dataMat.to_csv(os.path.join(ROOT_DIR, 'Data_Files/FoldDistributionsWSI.csv'))
#%%  TMA Data for paper
tmaCohorts={'TMA1':[],'TMA2':[],'PDX1':[]}

for cohort in tmaCohorts:
    responseInfo=PlUtils.Get_TMA_Response(cohort)
    scores=responseInfo.BAP1_Response.values
    gt=responseInfo.isBAP1WT.values
    patId=responseInfo.PatientID.values
    
    usedInAnalysis=np.logical_and(np.logical_and(np.isfinite(scores),
                                  np.isfinite(gt)),responseInfo.isBlackListed.values==False)
    
    numPatients=np.unique(patId[usedInAnalysis]).size
    totalSamples=np.sum(usedInAnalysis)
    numWt=np.sum(gt[usedInAnalysis]==1)
    numLoss=np.sum(gt[usedInAnalysis]==0)
    tmaCohorts[cohort]={'Patients':numPatients,'Punches': totalSamples,'Loss': numLoss,'WT': numWt}
    
    layoutFile=os.join.path(ROOT_DIR,projectPaths['Region_Level']['TMALayout'],cohort+'_Layout.csv')
    layout=pd.read(layoutFile)
    print('Original patients:',cohort, np.unique(layout.PatientID.values).size)