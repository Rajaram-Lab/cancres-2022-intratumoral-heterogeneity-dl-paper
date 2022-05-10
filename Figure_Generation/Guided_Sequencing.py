"""
This file is used to generate supplementary figuare 8b-c (guided sequencing plots).
The process is wrapped into functions so that it can be called through 
the FigureMasterScript.

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
import matplotlib.pyplot as plt
import yaml 
import sys

with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)

sys.path.insert(0, ROOT_DIR)
import Util.plotUtils as PlUtils
from tqdm import tqdm
# %% TMA Heterogeneity
def PlotTmaSequencing():
    tma1Response=PlUtils.Get_TMA_Response('TMA1',isNormalized=False,
                         showSlideImages=False,showPunchImages=False).set_index('PatientID')
    
    # Find Patients with Heterogeneity in True BAP1 Status
    uId=np.unique(tma1Response.index.values)
    hetPatients=[patientId for patientId in uId if 
                 len(np.unique(tma1Response.loc[[patientId]]['isBAP1WT']))>1]
    
    nCorrectModel=0
    for patientId in hetPatients:
        patientInfo=tma1Response.loc[[patientId]]
        punchMostLikelyBap1Loss=np.argmin(patientInfo['BAP1_Response'].values)
        isThatPunchBap1Loss=patientInfo.iloc[punchMostLikelyBap1Loss]['isBAP1WT']==False
        if isThatPunchBap1Loss:
            nCorrectModel+=1
    
    patientPunchStatus={}
    for patientId in hetPatients:
        patientPunchStatus[patientId]=tma1Response.loc[[patientId]]['isBAP1WT'].values
        
    numberOfRandomizations=int(1E6)        
    nCorrectRandom=[]    
    for randCounter in tqdm(range(numberOfRandomizations)):
        nCorrect=0
        for patientId in hetPatients:
            
            randomPunch=np.random.randint(len(patientPunchStatus[patientId]))
            isThatPunchBap1Loss=patientPunchStatus[patientId][randomPunch]==False
            if isThatPunchBap1Loss:
                nCorrect+=1
        nCorrectRandom.append(nCorrect)
    nCorrectRandom=np.array(nCorrectRandom)
    
    
    
    print('Model based selection is better than ',
          100*np.sum(nCorrectRandom<nCorrectModel)/numberOfRandomizations,
          '% of random samplings')
    
    corrCountsRandom=np.array([np.sum(nCorrectRandom==nCorr) for nCorr in range(len(hetPatients)+1)])
       
    # 
    plt.figure(figsize=(12,10))
    plt.bar(np.arange(len(hetPatients)+1),corrCountsRandom,
            color='r',alpha=0.4,width=1,edgecolor='k') 
    ylim=plt.ylim()
    plt.plot([nCorrectModel,nCorrectModel],ylim,'--r')
    ax1=plt.gca()
    ax1.set_ylabel('Frequency In Random Simulation')
    ax1.set_xlabel('# Patients (out of 7) where BAP1 Loss was identified')
    ax2=ax1.twinx()
    #cdf=np.concatenate([np.cumsum(corrCountsRandom)])/np.sum(corrCountsRandom)
    cdf=(np.concatenate([[0],np.cumsum(corrCountsRandom)])/np.sum(corrCountsRandom))[:-1]
    ax2.plot(np.arange(len(cdf)),cdf,label='Random Selection CDF')
    ax2.spines['right'].set_color('#1f77b4')
    ax2.spines['right'].set_linewidth(3)
    plt.ylim(0,1.05)
    # plt.legend(fontsize=14, loc=(0.01,0.94))
    ax2.set_ylabel('Cumulative Probability')
    plt.title('TMA1 Cohort')


# %%    Region Level Sampling from WSI
def PlotWsiSequencing():
    wsiRegionalPl=PlUtils.RegionalResponse('BAP1',100,0.7)
    trueClasses,predActivations,svsFiles=wsiRegionalPl.Get_WSI_Response(restrictToFocal=True)
    
    hetPatients=np.unique(svsFiles)
    
    nTopPicksToTest=3
    nCorrectModel=np.zeros(nTopPicksToTest)
    for patientId in hetPatients:
        isInPatient=svsFiles==patientId
        patientPredictions=predActivations[isInPatient]
        patientTrueClasses=trueClasses[isInPatient]
        areasMostLikelyBap1Loss=np.argsort(patientPredictions)
        areThoseAreasBap1Loss=patientTrueClasses[areasMostLikelyBap1Loss]==False
        nCorrectModel=nCorrectModel+areThoseAreasBap1Loss[:nTopPicksToTest]
            
    numberOfRandomizations=int(1E6)        
    nCorrectRandom=np.zeros(numberOfRandomizations)
    for randCounter in tqdm(range(numberOfRandomizations)):
    
        for patientId in hetPatients:
            isInPatient=svsFiles==patientId
            
            patientTrueClasses=trueClasses[isInPatient]
            randomArea=np.random.choice(len(patientTrueClasses))
            isRandomAreasBap1Loss=patientTrueClasses[randomArea]==False
            nCorrectRandom[randCounter]=nCorrectRandom[randCounter]+isRandomAreasBap1Loss
    
    for i in range(nTopPicksToTest):
        print('Model based selection Rank= '+ str(i+1)+'is better than ',
              100*np.sum(nCorrectRandom<nCorrectModel[i])/numberOfRandomizations,
              '% of random samplings')
    
    corrCountsRandom=[np.sum(nCorrectRandom==nCorr) for nCorr in range(len(hetPatients)+1)]
    
    #  Plot
    plt.figure(figsize=(12,10))
    plt.bar(np.arange(len(hetPatients)+1),corrCountsRandom,
            color='r',alpha=0.4,width=1,edgecolor='k') 
    ylim=plt.ylim()
    plt.plot([nCorrectModel[0],nCorrectModel[0]],ylim,'--r',label='Model Guided First Choice')
    plt.plot([nCorrectModel[1]-0.05,nCorrectModel[1]-0.05],ylim,'--b',label='Model Guided Second Choice')
    plt.plot([nCorrectModel[2],nCorrectModel[2]],ylim,'--g',label='Model Guided Third Choice')
    plt.legend(fontsize=14,loc='upper left')
    ax1=plt.gca()
    ax1.set_ylabel('Frequency In Random Simulation')
    ax1.set_xlabel('# Patients (out of 7) where BAP1 Loss was identified')
    ax2=ax1.twinx()
    cdf=(np.concatenate([[0],np.cumsum(corrCountsRandom)])/np.sum(corrCountsRandom))[:-1]
    ax2.plot(np.arange(len(cdf)),cdf,label='Random Selection CDF')
    ax2.spines['right'].set_color('#1f77b4')
    ax2.spines['right'].set_linewidth(3)
    plt.ylim(0,1.05)
    # plt.legend(loc=(0.01,0.9))
    plt.legend(fontsize=14, loc=(0.01,0.80))
    
    ax2.set_ylabel('Cumulative Probability')
    plt.title('WSI Cohort')
