#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to generate figure 5c and  supplementary figure 18 (grade 
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

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import scipy
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index

import pandas as pd 
import yaml

import sys
from matplotlib.patches import Patch


import statsmodels.stats.multitest as mt
from lifelines import CoxPHFitter

with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file) 

import Util.plotUtils as PlUtils

fontSize=figParams['fontsize']
mutColors=figParams['colors']
wtColors=figParams['wtColors']
geneList=figParams['geneList']

sys.path.insert(0, os.path.join(ROOT_DIR,'External/statannot/'))
from statannot.statannot import add_stat_annotation


# %% Grade Plots

def Plot_Grades(plotType,groupGrades,useStatAnnotPvals=True,isNormalized=True,fontSize=14):
 
    assert plotType in ['TMA2_Patient','TMA1_Punch']

    if plotType=='TMA2_Patient':
         cohortList=['TMA2']
         punchOrPatientLevel='patient'  
    elif plotType=='TMA1_Punch':
         cohortList=['TMA1']
         punchOrPatientLevel='punch'  
    else:
         sys.exit('Invalid Plot Type')
    
    assert punchOrPatientLevel in ['punch','patient']
    
    gradeDf=[]
    for cohort in cohortList:
        cohortResponse=PlUtils.Get_TMA_Response(cohort,isNormalized=isNormalized,
                             showSlideImages=False,showPunchImages=False).set_index('PatientID')
        if punchOrPatientLevel=='punch':
            assert cohort != 'TMA2', 'Punch level grades unavailable for TMA2'
            outDf=cohortResponse.copy()
        else:
            uId=np.unique(cohortResponse.index.values)
            idList=[]
            gradeList=[]
            bap1PredList=[]
            bap1TrueList=[]
            for patientId in uId:
                patientInfo=cohortResponse.loc[[patientId]]
                idList.append(patientId)
                # Represent patient by highest grade punch
                gradeList.append(np.max(patientInfo['Grade'].values))
                # Represent patient by punch with lowest activation (i.e. highest BAP1 likelihood) 
                bap1PredList.append(np.min(patientInfo['BAP1_Response'].values)) 
                #All punches need to be WT for patient to be WT
                bap1TrueList.append(np.all(patientInfo['isBAP1WT'].values)) 
            outDf=pd.DataFrame({'PatientID':idList,
                                'isBAP1WT':bap1TrueList,
                                'BAP1_Response':bap1PredList,
                                'Grade':gradeList}).set_index('PatientID')    
                
        outDf['Cohort']=[cohort]*outDf.shape[0]
        bap1Map={0:'Loss',1:'WT'}
        if groupGrades:
            gradeMap={1:'LowGrade',2:'LowGrade',3:'HighGrade',4:'HighGrade'}
            gradeOrder=['LowGrade_Loss','LowGrade_WT','HighGrade_Loss','HighGrade_WT']
            pValPairs=[('LowGrade_Loss','LowGrade_WT'),('HighGrade_Loss','HighGrade_WT')]
        else:
            gradeMap={1:1,2:2,3:3,4:4}
            gradeOrder=[]
            pValPairs=[]
            for grade in np.arange(1,5):
                gradeOrder=gradeOrder+[str(grade)+'_Loss',str(grade)+'_WT']
                pValPairs=pValPairs+[(str(grade)+'_Loss',str(grade)+'_WT')]
        outDf['GradeMapped']=[gradeMap[outDf.iloc[i]['Grade']] for i in range(outDf.shape[0])]
        outDf['BAP1']=[bap1Map[outDf.iloc[i]['isBAP1WT']] for i in range(outDf.shape[0])]
        
        outDf['GradexBap1']=[str(outDf.iloc[i]['GradeMapped'])+'_'+str(outDf.iloc[i]['BAP1']) \
         for i in range(outDf.shape[0])]
        gradeDf.append(outDf)
    
    gradeDf=pd.concat(gradeDf)
    
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='orange')
    b=sns.boxplot(data=gradeDf,x='GradexBap1',order=gradeOrder,\
                  y='BAP1_Response',hue='BAP1',hue_order=['Loss','WT'],
                  palette=[mutColors['BAP1'],wtColors['BAP1']],dodge=False,
                  showmeans=True,meanprops=meanlineprops,meanline=True)
    g=sns.swarmplot(data=gradeDf,x='GradexBap1',order=gradeOrder,\
                  y='BAP1_Response',hue='Cohort',
                  palette=['k','gray'],size=6,alpha=0.8,dodge=False)
    pVals=[]
    temp=gradeDf.set_index('GradexBap1')
    for pairs in pValPairs:
        if pairs[0] in temp.index and pairs[1] in temp.index:
            responseLoss=temp.loc[[pairs[0]]]['BAP1_Response'].values
            responseWT=temp.loc[[pairs[1]]]['BAP1_Response'].values
            u_stat,p=scipy.stats.mannwhitneyu(responseLoss,responseWT,alternative='two-sided') 
            pVals.append(p)
        else:
            pVals.append(1)
            
    correctedPvals=mt.multipletests(pvals=pVals,method='bonferroni')[1]
    
    if useStatAnnotPvals:
        add_stat_annotation(ax=b,data=gradeDf,x='GradexBap1',order=gradeOrder,\
                      y='BAP1_Response',
                            box_pairs=pValPairs,
                            text_format='simple',verbose=1,test='Mann-Whitney',
                            loc='outside',fontsize=fontSize)    
    else:
        add_stat_annotation(ax=b,data=gradeDf,x='GradexBap1',order=gradeOrder,\
                      y='BAP1_Response',
                            box_pairs=pValPairs,pvalues=correctedPvals,perform_stat_test=False,
                            text_format='simple',verbose=1,test=None,
                            loc='outside',fontsize=fontSize) 
        
      
    plt.yticks([0,1],['Confident\nLoss','Confident\nWT'],fontsize=fontSize)
    plt.ylim(0,1)
    
    g.set_xlabel("Grade",fontsize=fontSize)
    g.yaxis.set_label_coords(-0.025, 0.5)
    g.set_ylabel("Classifier Prediction",fontsize=fontSize)
    plt.setp(g.get_legend().get_texts(), fontsize='16') # for legend text
    plt.setp(g.get_legend().get_title(), fontsize='16') # for legend title
    g.tick_params(labelsize=14)
    
    legend_elements = [Patch(facecolor=mutColors['BAP1'],label= "Loss"),
                       Patch(facecolor=wtColors['BAP1'],label= "WT"),
                       Patch(facecolor='orange',label= "Mean")]
                      
    plt.legend(handles=legend_elements,fontsize=fontSize)
    if groupGrades:   
        plt.xticks([0.5,2.5],['Low (1-2)','High (3-4)'],fontsize=fontSize)
    else:
        plt.xticks([0.5,2.5,4.5,6.5],['1','2','3','4'])

    return gradeDf      


# %% Survival Analysis
def Plot_Survival(cohort='TMA2',survivalType='disease-specific',isNormalized=True,
                  returnValsOnly=False):    
    
    assert cohort in ['TMA2']
    assert survivalType in ['disease-specific']
  
    survivalFile=os.path.join(ROOT_DIR,projectPaths['Survival']['TMA2'])
    survivalInfo=pd.read_excel(survivalFile).set_index('Case_identifier')
    
    # Make Kaplan-Meier Curve NEED TO DETERMINE THESE FROM PAYAL/ALANA
    durationField='TIME_TO_Dead_Of_Disease'
    censoringField='Dead_Of_Disease'
    timeScaleFactor=1 # to convert time into months    

    
    # Load Response
    cohortResponse=PlUtils.Get_TMA_Response(cohort,isNormalized=isNormalized,
                         showSlideImages=False,showPunchImages=False).set_index('PatientID')
    cohortResponse.index = cohortResponse.index.map(int)
    # Get Patient Level Response
    uId=np.unique(cohortResponse.index.values)
    idList=[]
    bap1PredList=[]
    bap1TrueList=[]
    for patientId in uId:
        patientInfo=cohortResponse.loc[[patientId]]
        idList.append(patientId)
       
        # Represent patient by punch with lowest activation (i.e. highest BAP1 likelihood) 
        bap1PredList.append(np.min(patientInfo['BAP1_Response'].values)) 
        #All punches need to be WT for patient to be WT
        bap1TrueList.append(np.all(patientInfo['isBAP1WT'].values)) 
        
    survivalDf=pd.DataFrame({'PatientID':idList,
                        'isBAP1WT':bap1TrueList,
                        'BAP1_Response':bap1PredList}).set_index('PatientID')    
    durations=[]
    isCensored=[]
    for patientId in survivalDf.index:
        patientInfo=survivalInfo.loc[patientId]
        durations.append(patientInfo[durationField]*timeScaleFactor)    
        isCensored.append(patientInfo[censoringField]) 
        
    survivalDf['Duration']=durations
    survivalDf['isCensored']=isCensored
    
    # Supports two kinds of stratification by True or Predicted BAP1 Status
    stratification={'True_BAP1':survivalDf['isBAP1WT'].values,
                   'Pred_BAP1':survivalDf['BAP1_Response'].values>0.5}
    wtStatus={'WT':True,'Loss':False}
    # Styling
    stratLineStyle={'True_BAP1':'--','Pred_BAP1':'-'}
    stratLineWidth={'True_BAP1':1,'Pred_BAP1':3}
    statusColor={'WT':'#%02x%02x%02x' \
                 % tuple(np.uint8(np.round(np.array(wtColors['BAP1'])*255))),
                 'Loss':'#%02x%02x%02x' \
                 % tuple(np.uint8(np.round(np.array(mutColors['BAP1'])*255)))}
    fontSize=16
    
    discPvals={} # PValues
    discHR={} # Hazard Ratios
    for stratType in stratification: #Loop over stratification type
        coxDf=[]
        for count,status in enumerate(wtStatus): # Loop over BAP1 Loss/WT groups in stratification
            name=stratType+status
            # Make Kaplan-Meier Curve
            data=survivalDf[stratification[stratType]==wtStatus[status]].reset_index()
            kmf = KaplanMeierFitter()
            kmf.fit(data['Duration'], data['isCensored'],
                    label=name+ '(n='+str(data.shape[0])+')')
            if not returnValsOnly:
                ax=kmf.plot(ci_show=False,color=statusColor[status] ,
                            linestyle=stratLineStyle[stratType],show_censors=True,
                            linewidth=stratLineWidth[stratType]) 
                plt.xlabel('Months',fontsize=fontSize)
                plt.ylabel('Proportion of Patients',fontsize=fontSize)
                plt.legend(fontsize=fontSize,loc='upper right')
                locCoords={'TMA2_disease-specific':[5,0.63,0.6,1]}
                coordinateType=cohort+'_'+survivalType
                plt.ylim([locCoords[coordinateType][2],locCoords[coordinateType][3]])
                
            # Organize data for Cox Proportional Model p-value calculation
            data['status']=[wtStatus[status]]*data.shape[0]
            propHazardsDf=pd.DataFrame({'Duration':data['Duration'].values,
                                        'isCensored':data['isCensored'].values,
                                        'status':np.logical_not(data['status'].values)})
            coxDf.append(propHazardsDf)
            
        # Proprtional Hazards calculation based on BAP1 status        
        coxDf=pd.concat(coxDf)       
        cph = CoxPHFitter()
        cph.fit(coxDf, duration_col='Duration', event_col='isCensored')
        
        discHR[stratType]=cph.summary.loc['status']['exp(coef)'] 
        discPvals[stratType]=cph.summary.loc['status']['p']

    
    # Redefine prop hazards df without stratification
    propHazardsDf=pd.DataFrame({'Duration':survivalDf['Duration'].values,
                                'isCensored':survivalDf['isCensored'].values,
                                'BAP1_Response':1-survivalDf['BAP1_Response'].values})


    # Proportional Hazards calculation based on BAP1 status
    
    coxPH = CoxPHFitter()
    coxPH.fit(propHazardsDf, duration_col='Duration', event_col='isCensored')
    
    # Calculate concordance index without stratification 
    ci=[concordance_index(survivalDf['Duration'].values,
                         survivalDf['BAP1_Response'].values,
                         survivalDf['isCensored'].values),
        concordance_index(survivalDf['Duration'].values,
                         survivalDf['isBAP1WT'].values,
                         survivalDf['isCensored'].values)]

    cIndexPred=ci[0]
    cIndexTrue=ci[1]
    contPval=coxPH.summary.loc['BAP1_Response']['p']
    contHR=coxPH.summary.loc['BAP1_Response']['exp(coef)']
    
    contVars=[contHR,contPval,cIndexPred,cIndexTrue]
    discVars=[discHR, discPvals]
    if returnValsOnly:
        return contVars, discVars
