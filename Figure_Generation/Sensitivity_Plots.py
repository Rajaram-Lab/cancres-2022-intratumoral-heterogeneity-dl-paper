#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to generate figure 2e and  supplementary figure 10 (sensitivity 
plots). The process is wrapped into functions so that it can be called through 
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

with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
    

import Util.plotUtils as PlUtils

# Define plotting parameters 
fontSize=figParams['fontsize']
mutColors=figParams['colors']
wtColors=figParams['wtColors']
geneList=figParams['geneList']
       
#%%
def LoadData(classifier,norm,cohort='WSI'):
    """ Local function to load data 
    

    Parameters
    ----------
    classifier : str
        region or slide. Defines the spatial level of analysis for the model.
    norm : bool
        True or False. Use norm results for TMA
    cohort : str, optional
        WSI, TMA1, TMA2, PDX1. Cohort to load data from. The default is 'WSI'.

    Returns
    -------
    geneActivations : TYPE
        DESCRIPTION.
    geneIsWT : TYPE
        DESCRIPTION.
    geneIsFocal : TYPE
        DESCRIPTION.

    """

    if classifier=='slide':
        
        geneActivations,geneIsWT,geneIsFocal={},{},{}
        for gene in figParams['geneList']:
            predScore,isWt,isFocal,sampleList,scoresByFile=PlUtils.GetSlideLevelResults(cohort, gene)
            geneActivations[gene]=predScore
            geneIsWT[gene]=isWt
            geneIsFocal[gene]=isFocal
        
        
    elif classifier=='region':
        if cohort=='WSI':
            print("Plotting regional results")
            tessellationStride=figParams['RegionParams']['tessellationStride']
            minTumorThreshold=figParams['RegionParams']['threshold']
            gene='BAP1'
                
            regional=PlUtils.RegionalResponse(gene,tessellationStride,minTumorThreshold)
            geneIsWT, geneActivations, _=regional.Get_WSI_Response()
            geneActivations={'BAP1':geneActivations}
            geneIsWT={'BAP1':geneIsWT}
            # geneActivations,geneIsWT,_=PlUtils.Load_Region_Level_Results(figParams,calculate=False)
            geneIsFocal=None
        
        else:
            responseInfo=PlUtils.Get_TMA_Response(cohort=cohort,isNormalized=norm)
            if cohort=='TMA2':
                responseInfo=PlUtils.TMA_Patient_Level_Response(responseInfo)
                
            geneActivations={'BAP1':responseInfo.BAP1_Response.values}
            geneIsWT={'BAP1':responseInfo.isBAP1WT.values}
            geneIsFocal=None
            
    return geneActivations,geneIsWT, geneIsFocal


#%% Supplementary Figure 10: Complex Sensitivity plot for REGION classifier
    
def SensitivityComparison(saveResults=False):
    gene='BAP1'
    norm=True
    projects=[[figParams['cohortNames'][0],norm],[figParams['cohortNames'][1],norm]]  # True or False for Norm
    target=0.96
    # mix='NoMix'
    sensitivityLow=0.85 # lowest sensitivity to investigate (80 originally)
    plotStart=0.49  # 59 originally
    
    #Preload Region Results for more efficiency
    geneActivations,geneIsWT,geneIsFocal=LoadData(classifier='region',cohort='WSI',norm=False)
    
       
    isGood=np.logical_not(np.isnan(np.array(geneActivations[gene])))
    regionScores=np.array(geneActivations[gene])[isGood]
    regionStates=np.array(geneIsWT[gene])[isGood]
   
    regionOrder=np.argsort(regionScores)
    xVals=np.arange(len(regionOrder))/len(regionOrder)
    sensitivities,cutoffList,actualSensitivity,idx=[],[],[],[]
    
    #Calculate all sensitivities
    # This portion of the code finds all sensitivities under arbitrary cutoffs
    # defined by xVals. xVals is the same length as regionScores:
    for topScore in xVals:
        totMut=np.sum(np.logical_not(regionStates)) #all loss region call 
        mutUnderCutoff=np.sum(np.logical_not(regionStates[regionScores<topScore]))
        sensitivities.append(mutUnderCutoff/totMut)
    sensitivities=np.array(sensitivities)   # sensitivities using sorted activations
    
     
    #Isolate scores for target sensitivities
    # This loops over the desired range of sensitivies to plot. It shortens 
    # sensitivies to actualSensitivity and finds the activation cutoff at which
    # those sensitivies are met. Start of the range is senstivityLow
    sensRange=np.arange(sensitivityLow,1.0025,0.0025)
    for sen in sensRange:
        p=np.where(sensitivities>=sen)[0][0]
        idx.append(p)
        cutoffList.append(xVals[p])
        actualSensitivity.append(sensitivities[p])
    
    # Define and populate lists that will contain activations for all three 
    # projects: region WSI, TMA1 and TMA2 
    allGeneActs,allGeneStates=[],[]
    allGeneActs.append(regionScores)
    allGeneStates.append(regionStates)
    
    #Load TMA1 and TMA2 Results
    for project in projects:
        geneActivations,geneIsWT,geneIsFocal=LoadData(classifier='region',cohort=project[0], norm=project[1])
               
        isGood=np.logical_not(np.isnan(np.array(geneActivations[gene])))
        scores=np.array(geneActivations[gene])[isGood]
        states=np.array(geneIsWT[gene])[isGood]
      
        allGeneActs.append(scores)       
        allGeneStates.append(states)
                 
    # tmpPcntAll is the % of data examined for all three projects
    # tmpSensitivity all are the sensitivities for all three projects
    # the loop is set up to plot a desired range of sensitivities starting at 
    # plotStart
    tmpPcntAll=[]
    tmpSensitivityAll=[]
    for count in range(len(allGeneActs)):
        tmpPcnt=[]
        tmpSensitivity=[]
        for xVal in xVals[xVals>plotStart]:
            tmpPcnt.append(np.sum(allGeneActs[count]<xVal)/len(allGeneActs[count]))
            totMut=np.sum(np.logical_not(allGeneStates[count]))
            mutUnderCutoff=np.sum(np.logical_not(allGeneStates[count][allGeneActs[count]<xVal]))
            tmpSensitivity.append(mutUnderCutoff/totMut)
        tmpPcntAll.append(tmpPcnt)
        tmpSensitivityAll.append(tmpSensitivity)
        
        
    # Plotting functions    
    labels=['WSI Cohort','TMA1 Cohort','TMA2 Cohort']
    plt.figure(figsize=(15,15))
    ax1 = plt.subplot(221)
    # Plotting activation cutoff vs sensitivity
    for i in range(len(tmpSensitivityAll)):
        ax1.plot(xVals[xVals>=plotStart],tmpSensitivityAll[i], marker='o', label=labels[i],markersize =1)
    ax1.set_xlabel('Activations',fontsize=16)
    ax1.set_ylabel('Sensitivity',fontsize=16)
    ax1.set_xlim(plotStart,1)
    ax1.set_ylim(sensitivityLow-0.01,1.005)
    
    # Plotting vertical and horizontal dashed lines at points of interest -- can be optimized with loop
    senToPlot=[1,0.98,0.96,0.94,0.92]
    for x in senToPlot:
         plt.hlines(x, plotStart,cutoffList[np.where(np.array(actualSensitivity)>=x)[0][0]],linestyles='--')
         plt.vlines(cutoffList[np.where(np.array(actualSensitivity)>=x)[0][0]],0,x,linestyles='--')
    # Plotting red lines of interest 
    plt.hlines(target,0,cutoffList[np.where(np.array(actualSensitivity)>=target)[0][0]],linestyles='--',
               color='r')
    plt.vlines(cutoffList[np.where(np.array(actualSensitivity)>=target)[0][0]],0,target,linestyles='--',
               color='r')
    
    plt.legend(fontsize=16)
    
    #Plotting region WSI sensitivty vs region WSI, TMA1 and TMA2 
    labels=['Region-WSI','TMA1 Cohort','TMA2 Cohort']
    ax2 = plt.subplot(222,sharey=ax1)
    for j in range(len(labels)):
        ax2.plot(tmpSensitivityAll[0],tmpSensitivityAll[j],label=labels[j],marker='o',markersize =1)
    ax2.vlines(target,0,tmpSensitivityAll[1][np.where(np.array(tmpSensitivityAll[0])>=target)[0][0]],linestyles='--',
               label='Target Sensitivity/\nActivation',color='r')
    ax2.set_xlabel('WSI Sensitivity at Target Activations',fontsize=16)
    ax2.set_ylabel('Cohort Sensitivity at Targets Activation',fontsize=16)
    ax2.spines['bottom'].set_color('#1f77b4')
    ax2.spines['bottom'].set_linewidth(3)
    ax2.xaxis.label.set_color('#1f77b4')
    ax2.tick_params(axis='x', colors='#1f77b4')
    ax2.set_xlim(sensitivityLow,1)
    
    # Plot % Data excluded vs activations
    ax3=plt.subplot(223,sharex=ax1)
    for j in range(len(labels)):
        ax3.plot(xVals[xVals>=plotStart],np.array(tmpPcntAll[j])*100, marker='o',label=labels[j],markersize =1)
    ax3.set_xlabel('Activations',fontsize=16)
    ax3.set_ylabel('% Data Examined',fontsize=16)
    ax3.set_ylim(20,102)
    
    # Plot % data excluded vs region WSI sensitivity
    ax4= plt.subplot(224,sharex=ax2,sharey=ax3)
    for j in range(len(labels)):
        ax4.plot(tmpSensitivityAll[0],np.array(tmpPcntAll[j])*100,label=labels[j],marker='o',markersize =1)
    ax4.vlines(target,0,(tmpPcntAll[1][np.where(np.array(tmpSensitivityAll[0])>=target)[0][0]])*100,linestyles='--',
               label='Target Sensitivity/\nActivation',color='r')
    ax4.set_xlabel('WSI Sensitivity at Target Activations',fontsize=16)
    ax4.set_ylabel('% Data Examined',fontsize=16)
    ax4.spines['bottom'].set_color('#1f77b4')
    ax4.spines['bottom'].set_linewidth(3)
    ax4.xaxis.label.set_color('#1f77b4')
    ax4.tick_params(axis='x', colors='#1f77b4')
    
    plt.tight_layout()
# SensitivityComparison()

#%% Fig 2e Plot: This is the large sensitivity plot for the figure.

def SensitivityPlot():
    gene='BAP1'
    projects=[['slide',True,'WSI','-'],
              ['slide',True,'TCGA','--']]  # True or False for Norm
    
    target=0.96
    
    plt.figure(figsize=(16,13))
    
    # Plotting top portion of the plot
    h=plt.subplot2grid((12,1),(0,0),rowspan=8)
    #Preload slide Results for more efficiency
    # projectData={}
    sensitivityValsToReturn={}
    for count,project in enumerate(projects):
        geneActivations,geneIsWT,geneIsFocal=LoadData(classifier=project[0],
                                                      norm=project[1],cohort=project[2])
    
        isGood=np.logical_not(np.isnan(np.array(geneActivations[gene])))
        scores=np.array(geneActivations[gene])[isGood]
        states=np.array(geneIsWT[gene])[isGood]
        
        try:
            isFocalGood=geneIsFocal[gene][isGood]
        except TypeError:
            print('No focal data found')
        order=np.argsort(scores)
        xVals=np.arange(len(order))/len(order)
        sensitivities=[]
        # cutoffList,actualSensitivity,idx=[],[],[]
        
        #Calculate all sensitivities from a data driven point. Cutoffs for 
        # activation are selected based on sorted activations
        for topScore in scores[order]:
            totMut=np.sum(np.logical_not(states))
            mutUnderCutoff=np.sum(np.logical_not(states[scores<topScore]))
            sensitivities.append(mutUnderCutoff/totMut)
        sensitivities=np.array(sensitivities)
        
        
        validWTSamples=np.array(np.logical_and(np.isfinite(geneActivations[gene]),geneIsWT[gene]==1),dtype=np.uint8)
        validMutSamples=np.array(np.logical_and(np.isfinite(geneActivations[gene]),geneIsWT[gene]==0),dtype=np.uint8)
        
        plt.step(xVals,sensitivities,markersize=2,where='post',color=mutColors[gene],
                 linestyle=project[-1],
                 label=project[2]+': WT='+str(np.sum(validWTSamples))+\
                 ' Loss='+str(np.sum(validMutSamples)))
        
        t=np.where(sensitivities>=target)[0][0]
        plt.hlines(target,0,xVals[t], color='gray',linestyle='--')
        plt.vlines(xVals[t],0,target, color='gray',linestyle='--')
        # print(xVals[t])
        plt.ylabel('Proportion of BAP1 Loss Captured',fontsize=16)
        plt.xlabel('Proportion of Data Considered',fontsize=16)
        plt.ylim([0,1.01])
        plt.xlim([0,1])
        plt.title('Sensitivity',fontsize=16)
        plt.legend(fontsize=16)
        
        sensitivityValsToReturn[project[2]]=[xVals[t],
                                             xVals[np.where(sensitivities>=1)[0][0]]]
        
    h=plt.subplot2grid((12,1),(8,0),rowspan=2)
    for count,project in enumerate(projects):
        geneActivations,geneIsWT,geneIsFocal=LoadData(classifier=project[0],
                                                      norm=project[1],cohort=project[2])
    
        isGood=np.logical_not(np.isnan(np.array(geneActivations[gene])))
        scores=np.array(geneActivations[gene])[isGood]
        states=np.array(geneIsWT[gene])[isGood]
        
        try:
            isFocalGood=geneIsFocal[gene][isGood]
        except TypeError:
            print('No focal data found')
        order=np.argsort(scores)
        xVals=np.arange(len(order))/len(order)
        plt.plot(scores[order],color=mutColors[gene],linestyle=project[-1])
    
        plt.xticks([],[])
        xLim=plt.xlim(0,len(xVals))    
        plt.ylabel('Activation',fontsize=fontSize)
        h.yaxis.tick_right() 
    
    
    plt.subplot2grid((12,1),(10,0),rowspan=1)
    project=projects[0]
    geneActivations,geneIsWT,geneIsFocal=LoadData(classifier=project[0],
                                                      norm=project[1],cohort=project[2])
    
    isGood=np.logical_not(np.isnan(np.array(geneActivations[gene])))
    scores=np.array(geneActivations[gene])[isGood]
    states=np.array(geneIsWT[gene])[isGood]
    
    try:
        isFocalGood=geneIsFocal[gene][isGood]
    except TypeError:
        print('No focal data found')
    order=np.argsort(scores)
    xVals=np.arange(len(order))/len(order)
    stateMat=np.uint8(states[order][np.newaxis])
    try:
        stateMat[0,isFocalGood[order]]=2
    except UnboundLocalError:
        print('No focal ticks plotted')

    negIdx=np.where(stateMat[0,:]!=1)[0]
    for i in negIdx:
        if stateMat[0,i]==0:
            c=mutColors[gene]
        else:
            c='orange'
        plt.vlines(i,0,1,color=c,linewidth=1,linestyle=project[-1])    
    
    plt.ylim(0,1)    
    plt.xlim(0,len(xVals)-1)
    plt.yticks([],[])
    plt.ylabel(project[2],fontsize=fontSize)  
    
    plt.subplot2grid((12,1),(11,0),rowspan=1)
    project=projects[1]
    geneActivations,geneIsWT,geneIsFocal=LoadData(classifier=project[0],
                                                      norm=project[1],cohort=project[2])
    
    isGood=np.logical_not(np.isnan(np.array(geneActivations[gene])))
    scores=np.array(geneActivations[gene])[isGood]
    states=np.array(geneIsWT[gene])[isGood]
    
    try:
        isFocalGood=geneIsFocal[gene][isGood]
    except TypeError:
        print('No focal data found')
    order=np.argsort(scores)
    xVals=np.arange(len(order))/len(order)
    stateMat=np.uint8(states[order][np.newaxis])
    try:
        stateMat[0,isFocalGood[order]]=2
    except UnboundLocalError:
        print('No focal ticks plotted')
    #plt.imshow(stateMat,aspect='auto',cmap=ListedColormap([geneColors['BAP1'],'w','orange']))
    negIdx=np.where(stateMat[0,:]!=1)[0]
    for i in negIdx:
        if stateMat[0,i]==0:
            c=mutColors[gene]
        else:
            c='orange'
        plt.vlines(i,0,1,color=c,linewidth=1,linestyle=project[-1])    
    plt.ylim(0,1)    
    plt.xlim(0,len(xVals)-1)
    plt.yticks([],[])
    plt.xlabel('Samples sorted by confidence of being '+gene+' WT',fontsize=fontSize)
    plt.ylabel(project[2],fontsize=fontSize)  
    
    
    plt.tight_layout(pad=0.01)
    return sensitivityValsToReturn
# SensitivityPlot()


#%% Supplementary Figure 6: Sensitivity plots

# This function is called by ConfidenceSentivityPlot below
def PlotSimpleSensitivity(geneActivations,geneIsWT,geneIsFocal,gene):
    plt.figure(figsize=(10,13))
    h=plt.subplot2grid((12,1),(0,0),rowspan=8)
    
    isGood=np.logical_not(np.isnan(np.array(geneActivations[gene])))
    scores=np.array(geneActivations[gene])[isGood]
    states=np.array(geneIsWT[gene])[isGood]
    
    try:
        isFocalGood=geneIsFocal[gene][isGood]
    except TypeError:
        pass 
        
    order=np.argsort(scores)
    xVals=np.arange(len(order))/len(order)
    sensitivities=[]
        
    #Calculate all sensitivities
    for topScore in scores[order]:
        totMut=np.sum(np.logical_not(states))
        mutUnderCutoff=np.sum(np.logical_not(states[scores<topScore]))
        # print(mutUnderCutoff)
        sensitivities.append(mutUnderCutoff/totMut)
    sensitivities=np.array(sensitivities)
    
     
    #Isolate scores  target sensitivities
  
    sensitivitiesToReturn=[]
    t1=np.where(sensitivities>=0.96)[0][0]
    t2=np.where(sensitivities>=1)[0][0]
    sensitivitiesToReturn=[xVals[t1],xVals[t2]]
    
    plt.plot(xVals,sensitivities,marker='o',markersize=2,label='Data: WT='+str(int(np.sum(geneIsWT[gene])))+\
             ', Loss='+str(int((geneIsWT[gene].size-np.sum(geneIsWT[gene])))),color=mutColors[gene])
    plt.ylabel('Proportion of BAP1 Loss Captured',fontsize=16)
    plt.xlabel('Proportion of Data Considered',fontsize=16)
    plt.ylim([0,1.01])
    plt.xlim([0,1])
    
    plt.title('Sensitivity',fontsize=16)
    plt.legend(fontsize=16)
        
    h=plt.subplot2grid((12,1),(8,0),rowspan=2)
       
    try:
        isFocalGood=geneIsFocal[gene][isGood]
    except TypeError:
        pass
    
    order=np.argsort(scores)
    xVals=np.arange(len(order))/len(order)
    plt.plot(scores[order],color=mutColors[gene])
    
    plt.xticks([],[])
    xLim=plt.xlim(0,len(xVals))    
    plt.ylabel('Activation',fontsize=fontSize)
    h.yaxis.tick_right() 
    
    plt.subplot2grid((12,1),(10,0),rowspan=1)
    
    try:
        isFocalGood=geneIsFocal[gene][isGood]
    except TypeError:
        pass
    
    order=np.argsort(scores)
    xVals=np.arange(len(order))/len(order)
    stateMat=np.uint8(states[order][np.newaxis])
    try:
        stateMat[0,isFocalGood[order]]=2
    except UnboundLocalError:
        pass
    
    negIdx=np.where(stateMat[0,:]!=1)[0]
    for i in negIdx:
        if stateMat[0,i]==0:
            c=mutColors[gene]
        else:
            c='orange'
        plt.vlines(i,0,1,color=c,linewidth=1)    
    
    plt.ylim(0,1)    
    plt.xlim(0,len(xVals))
    plt.yticks([],[])
    plt.ylabel(gene,fontsize=fontSize)  
    plt.xlabel('Samples sorted by confidence of being '+gene+' WT',fontsize=fontSize)
    plt.tight_layout(pad=0.01)
    return sensitivitiesToReturn

