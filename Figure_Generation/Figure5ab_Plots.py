"""
This file is used to generate figure 5a,b (classifation independence plot for BAP1
and PBRM1 and Nuclear feature difference for BAP1 and PBRM1) and supplementary
figure 12 (gene independence for BAP1 and SETD2, and PBRM1 and SETD2).
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


import os, sys
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir
sys.path.insert(0, ROOT_DIR) 

import yaml

import seaborn as sns
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import rankdata
import pandas as pd 
from matplotlib.patches import Patch
from matplotlib import cm
# %matplotlib inline


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
## %% Fig 5a: Independence of gene classifiers
def ClassificationIndependence(genePair,figPlot=False,plotNum=None):
    """

    Parameters
    ----------
    genePair : TYPE
        DESCRIPTION.
    figPlot : TYPE, optional
        DESCRIPTION. The default is False.
    plotNum : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    #ClassificationIndependence
    gene1,gene2=genePair
    hueColors=['lightgray',mutColors[gene1],mutColors[gene2],'magenta']
    cohort='WSI'
    # geneActivations,geneIsWT,geneIsFocal,_=PlUtils.Load_Slide_Level_Results()    
    geneActivations,geneIsWT,geneIsFocal={},{},{}
    for gene in figParams['geneList']:
        predScore,isWt,isFocal,_,_=PlUtils.GetSlideLevelResults(cohort, gene)
        geneActivations[gene]=predScore
        geneIsFocal[gene]=isFocal
        geneIsWT[gene]=isWt
        
    isGood=np.logical_and(np.logical_not(geneIsFocal[gene1]),
                          np.logical_not(geneIsFocal[gene2]))
    isGood=np.logical_and(isGood,np.logical_not(np.logical_or(\
                        np.isnan(geneActivations[gene1]),\
                            np.isnan(geneActivations[gene2]))))
        
        
    geneState=(1-np.int16(geneIsWT[gene1]))+2*(1-np.int16(geneIsWT[gene2]))
    # 0:both genes WT, 1:gene1=Loss,gene2=WT,2:gene1=WT,gene2=Loss;3:both Loss
    
    
    X=rankdata(1-geneActivations[gene1][isGood])/np.sum(isGood)
    Y=rankdata(1-geneActivations[gene2][isGood])/np.sum(isGood)
    hueClass=geneState[isGood]
    if not figPlot:
        legend_elements = [Patch(facecolor='white',label=gene1+'  |  '+gene2),
                           Patch(facecolor=hueColors[0],label=' WT    |   WT'),
                           Patch(facecolor=hueColors[1],label=' Loss  |   WT'),
                           Patch(facecolor=hueColors[2],label=' WT    |   Loss'),
                           Patch(facecolor=hueColors[3],label=' Loss  |   Loss')]
                
        f, (a0, a1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1,3]},figsize=(10,16))
        for hC in range(4): 
            sns.distplot(X[hueClass==hC],ax=a0,
                      color=hueColors[hC],norm_hist=True,kde=True,hist=False,rug=True)
        a0.set_xlim(0,1)    
        a0.set_ylabel('Kernel Density',fontsize=fontSize)
        a0.set_xlabel(gene1+' Loss Ranking',fontsize=fontSize)
        a0.legend(handles=legend_elements, loc='upper left',fontsize=fontSize)
        a1.scatter(X,Y,200,hueClass,cmap=ListedColormap(hueColors))
        a1.set_xlim(0,1)  
        a1.set_ylim(0,1)  
        #a1.axis('square')
        a1.set_xlabel(gene1+' Loss Ranking',fontsize=fontSize)
        a1.set_ylabel(gene2+' Loss Ranking',fontsize=fontSize)
    else: 
        if plotNum==0:
            legend_elements = [Patch(facecolor='white',label=gene1+'  |  '+gene2),
                           Patch(facecolor=hueColors[0],label=' WT    |   WT'),
                           Patch(facecolor=hueColors[1],label=' Loss  |   WT'),
                           Patch(facecolor=hueColors[2],label=' WT    |   Loss'),
                           Patch(facecolor=hueColors[3],label=' Loss  |   Loss')]
            for hC in range(4):
                sns.distplot(X[hueClass==hC],
                          color=hueColors[hC],norm_hist=True,kde=True,hist=False,rug=True)
                plt.xlim(0,1)
                plt.xlabel(gene1+' Loss Ranking',fontsize=fontSize)
                plt.ylabel('Kernel Density',fontsize=fontSize)
                plt.legend(handles=legend_elements, loc='upper left',fontsize=fontSize)
        elif plotNum==1: 
            plt.scatter(X,Y,100,hueClass,cmap=ListedColormap(hueColors))
            plt.xlabel(gene1+' Loss Ranking',fontsize=fontSize)
            plt.ylabel(gene2+' Loss Ranking',fontsize=fontSize)
            plt.xlim(0,1)
            plt.ylim(0,1)
    
# ClassificationIndependence(['PBRM1','SETD2'])
#%% Figure 5c: Nuclear feature density plots for BAP1 & PBRM1

     
def NuclearFeatureDensity(plotNum=None):
    """IMPORTANT: This function currently works when called into the figure 5 
    plot. After removing figplot arm, it doesnt work when called individually

    Parameters
    ----------

    figPlot : TYPE, optional
        DESCRIPTION. The default is False.
    plotNum : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    featureFile=os.path.join(projectPaths['DataDir'],
                             projectPaths['Nuclear']['CombinedFeats'])
    
    [combinedFeatMat,svsList,allFeatNames]=pickle.load(open(featureFile,'rb'))
    
    allSampleFile=os.path.join(ROOT_DIR,projectPaths['Data']['AllSamples'])
    
    allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1).set_index('svs')
        
    
    isBAP1Pos=allSamples.loc[svsList]['BAP1_Positive'].values
    isPBRM1Pos=allSamples.loc[svsList]['PBRM1_Positive'].values
    
    
    feat1=allFeatNames.index('Avg_BBox_Area')
    feat2=allFeatNames.index('Avg_H_Int_Min')

    if plotNum==0:
        legend_elements=[Patch(facecolor=[0,0.75,0],label='WT'),
                     Patch(facecolor=[0.75,0,0],label='Loss')]
        isGenePos=isBAP1Pos
        gene='BAP1'
        WtFeatMat=combinedFeatMat[isGenePos==1,:]
        MutFeatMat=combinedFeatMat[isGenePos==0,:]
        sns.kdeplot(WtFeatMat[:,feat1],WtFeatMat[:,feat2],cmap=cm.get_cmap('Greens', 128),shade=False,shade_lowest=False)
        sns.kdeplot(MutFeatMat[:,feat1],MutFeatMat[:,feat2],cmap=cm.get_cmap('Reds', 128), shade=False,shade_lowest=False)
        plt.xlabel('Nuclear Area',fontsize=fontSize)
        plt.ylabel('Hematoxylin Nuclear\nIntensity',fontsize=fontSize)
        plt.text(25,-0.555,'Lighter',fontsize=12)
        plt.text(25,-0.465,'Darker',fontsize=12)
        plt.title(gene,fontsize=fontSize)
        plt.legend(handles=legend_elements,fontsize=fontSize)
    elif plotNum==1:
        isGenePos=isPBRM1Pos
        gene='PBRM1'
        WtFeatMat=combinedFeatMat[isGenePos==1,:]
        MutFeatMat=combinedFeatMat[isGenePos==0,:]
        sns.kdeplot(WtFeatMat[:,feat1],WtFeatMat[:,feat2],cmap='Greens',shade=False,shade_lowest=False)
        sns.kdeplot(MutFeatMat[:,feat1],MutFeatMat[:,feat2],cmap='Reds',shade=False,shade_lowest=False)
        plt.xlabel('Nuclear Area',fontsize=fontSize)
        plt.ylabel('Hematoxylin Nuclear\nIntensity',fontsize=fontSize)
        plt.text(24.85,-0.5605,'Lighter',fontsize=12)
        plt.text(24.85,-0.465,'Darker',fontsize=12)
        plt.title(gene,fontsize=fontSize)
        
# plt.figure(figsize=(8,10))
# NuclearFeatureDensity(plotNum=1) # featuers that you are interested in visualizing; can be hardcoded  [0,3,11,10] - [0,11] [3,10]
            




    