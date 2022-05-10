"""
GradeVsMutation contains the functions to regenerate the plots b-d found in
supplementary figure 5. This function will be called in the Figure_Generation 
python notebook.

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
import pandas as pd
import matplotlib.pyplot as plt
import Util.plotUtils as plUtils
import yaml 

with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
#%%

def Plot_GradeVsMutation(gene, projectPaths, verbose=0):
    """
    This function will plot the effect of prediction mutations within low and
    high grades.

    Parameters
    ----------
    gene : string
        Gene to analyze. BAP1, PBRM1 or SETD2
    projectPaths : dict
        Dictionary containing all project paths.
    verbose : int
        Specify whether to print warning messages. Default is 0 for no messages
    Returns
    -------
    None. Plots AUC analysis 

    """
    cohort = 'TCGA'
    tcgaGradeFile= projectPaths['Slide_Level']['GradeTCGA']
    df = pd.read_csv(os.path.join(ROOT_DIR, tcgaGradeFile))
    
    predScore, isWt, isFocal, sampleList, scoresByFile = plUtils.GetSlideLevelResults(cohort, gene)


    filter_predScoreIsValid, filter_gradeType = [], []
    for sampleIx, sample in enumerate(sampleList):
        if np.isnan(predScore[sampleIx]):
            filter_predScoreIsValid.append(False)
        else:
            filter_predScoreIsValid.append(True)
        
        gradesListed = df[df['patient'] == sample]['g-grade'].tolist()
        if len(gradesListed) == 0:
            if verbose == 1:
                print(f"No occurrences for sample {sample}")
            filter_gradeType.append(0)
        elif len(gradesListed) > 1:
            if verbose == 1:
                print(f"Mutiple occurences for {sample}")
            filter_gradeType.append(0)
        elif len(gradesListed) == 1:
            if gradesListed[0].strip() in ['G1', 'G2']:
                filter_gradeType.append(1)
            elif gradesListed[0].strip() in ['G3', 'G4']:
                filter_gradeType.append(2)
            else:
                filter_gradeType.append(0)
    
    filter_predScoreIsValid = np.array(filter_predScoreIsValid)
    filter_gradeType = np.array(filter_gradeType)
    
    filter_lowGrade = filter_gradeType == 1
    filter_highGrade = filter_gradeType == 2
    filter_validGrade = filter_gradeType != 0
    
    numWT_lowGrade, numLoss_lowGrade = None, None
    vals, counts = np.unique(isWt[filter_predScoreIsValid & filter_lowGrade], return_counts=True)
    for val, cnt in zip(vals, counts):
        if val == True:
            numWT_lowGrade = cnt
        elif val == False:
            numLoss_lowGrade = cnt
    
    
    numWT_highGrade, numLoss_highGrade = None, None
    vals, counts = np.unique(isWt[filter_predScoreIsValid & filter_highGrade], return_counts=True)
    for val, cnt in zip(vals, counts):
        if val == True:
            numWT_highGrade = cnt
        elif val == False:
            numLoss_highGrade = cnt
    
    
    numWT_allGrade, numLoss_allGrade = None, None
    vals, counts = np.unique(isWt[filter_predScoreIsValid & filter_validGrade], return_counts=True)
    for val, cnt in zip(vals, counts):
        if val == True:
            numWT_allGrade = cnt
        elif val == False:
            numLoss_allGrade = cnt
            
    aucResultsList=[]
    colorsList=['limegreen','red','blue']
    labelsList=[f'Low Grade (WT:{numWT_lowGrade}, Loss:{numLoss_lowGrade})',
                f'High Grade (WT:{numWT_highGrade}, Loss:{numLoss_highGrade})',
                f'All Grade (WT:{numWT_allGrade}, Loss:{numLoss_allGrade})']
    
    plt.figure(figsize=(10, 10))
    aucResultsList.append(plUtils.ROC_Plot(title=f'{gene}', 
        truePosLabels=np.uint8(isWt[filter_predScoreIsValid & filter_lowGrade]), predictedPctPositive=predScore[filter_predScoreIsValid & filter_lowGrade], 
        color='limegreen', geneName=gene, plot=False))
    
    aucResultsList.append(plUtils.ROC_Plot(title=f'{gene}', 
        truePosLabels=np.uint8(isWt[filter_predScoreIsValid & filter_highGrade]), predictedPctPositive=predScore[filter_predScoreIsValid & filter_highGrade], 
        color='red', geneName=gene, plot=False))
    
    aucResultsList.append(plUtils.ROC_Plot(title=f'{gene}', 
        truePosLabels=np.uint8(isWt[filter_predScoreIsValid & filter_validGrade]), predictedPctPositive=predScore[filter_predScoreIsValid  & filter_validGrade], 
        color='blue', geneName=gene, plot=False))
    
    for i,aucResults in enumerate(aucResultsList):
        plt.plot(aucResults[1],aucResults[2],color=colorsList[i],
             label=labelsList[i]+' (AUC = %0.2f)' % aucResults[0]) 

        
        plt.xlabel('False Positive Rate',fontsize=16)
        plt.ylabel('True Positive Rate',fontsize=16)
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')  
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc="lower right",fontsize=16)
        plt.axis('square')  
        plt.title(gene,fontsize=16)