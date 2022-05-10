#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to generate all the AUC plots for multiple figures across the
paper. The process is wrapped into functions so that it can be called through 
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

# %matplotlib inline
# import subprocess 

# ROOT_DIR=subprocess.getoutput("git rev-parse --show-toplevel")
with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)

import Util.plotUtils as PlUtils
fontSize=figParams['fontsize']
mutColors=figParams['colors']
wtColors=figParams['wtColors']
geneList=figParams['geneList']


# %% Fig. 2c/2d: Slide level AUC curves for all 3 genes

    
def PlotWsiSlideAUCs(cohort,missenseAsWT=False):
    """
    

    Parameters
    ----------
    mix : TYPE
        DESCRIPTION.
    cohort : TYPE
        DESCRIPTION.
    figPlot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    plt.figure(figsize=(10,10))
     
    for gene in ['BAP1','PBRM1','SETD2']:
        geneActivations,geneIsWT,geneIsFocal,sampleList,scoreByFile=PlUtils.GetSlideLevelResults(cohort, gene,
                                                                                                 isMisSenseTreatedAsWT=missenseAsWT)
        
        # mutColors={'BAP1':[0.75,0,0],'PBRM1':[0,0,0.75],'SETD2':[0,0.75,0.75]}
        aucResults=PlUtils.ROC_Plot(np.uint8(geneIsWT),geneActivations,
                                   mutColors[gene],gene,plot=False)   
        plt.plot(aucResults[1],aucResults[2],color=mutColors[gene],
             label=gene+'(AUC = %0.2f)' % aucResults[0]) 
    
        
        plt.xlabel('False Positive Rate',fontsize=16)
        plt.ylabel('True Positive Rate',fontsize=16)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right",fontsize=16)
        plt.axis('square')   
    
    
# PlotWsiSlideAUCs(cohort='TCGA',missenseAsWT=True)


#%% Regional ROC curves

def PlotWsiRegionAUCs():
    """
    

    Parameters
    ----------
    calculate : TYPE, optional
        DESCRIPTION. The default is False.
    figPlot : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    tessellationStride=figParams['RegionParams']['tessellationStride']
    minTumorThreshold=figParams['RegionParams']['threshold']
    plt.figure(figsize=(10,10))
        
    for gene in geneList:
        regional=PlUtils.RegionalResponse(gene,tessellationStride,minTumorThreshold)
        trueClasses, predActivations, svsFiles=regional.Get_WSI_Response()
        
        auc_score,fpr,tpr=PlUtils.ROC_Plot(np.array(trueClasses),
                                    np.array(predActivations),mutColors[gene],gene)
        
# PlotWsiRegionAUCs()
#%% TMA AUC Calculations

def PlotTmaAUCs(geneName='BAP1',includedTMA='all',patientLevel=False,norm=True):
    """
    

    Parameters
    ----------

    geneName : TYPE, optional
        DESCRIPTION. The default is 'BAP1'.
    includedTMA : TYPE, optional
        'all' will display the corresponding AUCs for all three TMA cohorts - TMA1,TMA2,
        PDX1 in a single plot. 'main' will display only the AUCs for TMA1 and PDX1. 
        'single' will plot all AUCs in seperate plots. The default is 'all'.
    norm : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    
    projectNames=[[figParams['cohortNames'][0],norm],[figParams['cohortNames'][1],norm],
                  [figParams['cohortNames'][2],norm]]
    linestyles=['-','--','-.']
    lineColors=[mutColors[geneName],'darkred','lightcoral']
   
    responses=[]
    for project in projectNames:
        if project[0]=='TMA2':
            punchLevelResponse=PlUtils.Get_TMA_Response(cohort=project[0],isNormalized=project[1])
            responses.append(PlUtils.TMA_Patient_Level_Response(punchLevelResponse))
        else:
            if not patientLevel:
                responses.append(PlUtils.Get_TMA_Response(cohort=project[0],isNormalized=project[1]))
            else:
                responses.append(PlUtils.TMA_Patient_Level_Response(PlUtils.Get_TMA_Response(cohort=project[0],isNormalized=project[1])))
            
       
    if includedTMA=='all':
        plt.figure(figsize=(10,10))
        for i,response in enumerate(responses):
           
            auc_score,fpr,tpr=PlUtils.ROC_Plot(response.isBAP1WT.values,response.BAP1_Response.values,
                                   lineColors[i],geneName,ls=linestyles[i],project=projectNames[i][0]) 
    elif includedTMA=='main':
        plt.figure(figsize=(10,10))
        for i,response in enumerate(responses):
            if i!=1:
                auc_score,fpr,tpr=PlUtils.ROC_Plot(response.isBAP1WT.values,response.BAP1_Response.values,
                                   lineColors[i],geneName,ls=linestyles[i],project=projectNames[i][0])  

        
# PlotTmaAUCs(includedTMA='all',patientLevel=False, norm=False)

#%% Grade-based classification auc
def CalculateGradeAUC(plot,cohort='TMA1'):
    gene='BAP1'
    gradeData=PlUtils.Get_TMA_Response(cohort=cohort,isNormalized=True)
    if cohort == 'TMA2':
        gradeData=PlUtils.TMA_Patient_Level_Response(gradeData)
    normGrades=gradeData.Grade.values/5
    if plot:
        plt.figure(figsize=(10,10))
    aucResults=PlUtils.ROC_Plot(np.uint8(gradeData['isBAP1WT'].values),1-normGrades,
                                       mutColors['BAP1'],gene,plot=plot) 
    return aucResults[0]
# CalculateGradeAUC(cohort='TMA2',plot=False)