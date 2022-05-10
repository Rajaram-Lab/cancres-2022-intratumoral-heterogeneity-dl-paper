"""
This file is used to generate figure 4a,c (activation plots for TMA1 and PDX1).
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

import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

import yaml

with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file) 


dataDir=projectPaths['DataDir']

from matplotlib import cm

# %%
def Plot_TMA_ActivationMaps():
    bottom = cm.get_cmap('Greens', 128)
    top = cm.get_cmap('Reds', 128)
    newcolors = np.vstack((top(np.array(list(reversed(np.linspace(0,1,128))))),
                            bottom(np.linspace(0, 1, 128))))
    gwrCmap = ListedColormap(newcolors, name='GreenRed')
        
   
    activationsDir=os.path.join(dataDir,projectPaths['ResponseData'],
                                'Region_Level/TMA_NoNorm/')
    layoutDir=os.path.join(ROOT_DIR,projectPaths['Region_Level']['TMALayout'])
    cohortList=['TMA1','PDX1']
    dsf=[1,2]
    tessellationStride=100
    
    suffix='_avg.pkl'
    
    for i,cohort in enumerate(cohortList):
    
    
        layoutFile=os.path.join(layoutDir,cohort+'_Layout.csv')
        fullLayout=pd.read_csv(layoutFile).set_index('SVS')
        uniqueSvs=np.unique(fullLayout.index)
        for svs in uniqueSvs[:1]:
            
            svsLayout=fullLayout.loc[svs]
           
            avgActivationFile=os.path.join(
                activationsDir,svs.replace('.svs',suffix))
            avgAct=pickle.load(open(avgActivationFile,'rb'))           
            if suffix=='_avg.pkl':
                avgAct=avgAct[1][:,:,1]
            else:                       
                avgAct=avgAct[:,:,1]
            
            fig=plt.figure(figsize=(20,10))        
            plt.imshow(avgAct,vmin=-0.2,vmax=1.2,cmap=gwrCmap)
            for punchCounter in range(svsLayout.shape[0]):
                punchInfo=svsLayout.iloc[punchCounter]
                cornerPos=(punchInfo.CornerX/(tessellationStride*dsf[i]),
                           punchInfo.CornerY/(tessellationStride*dsf[i]))
                rect=np.array([[0,0],[0,1],[1,1],[1,0],[0,0]])
                rect[:,0]=punchInfo.Height*rect[:,0]/(tessellationStride*dsf[i])
                rect[:,1]=punchInfo.Width*rect[:,1]/(tessellationStride*dsf[i])
                rect=rect+np.array(cornerPos)
                isGood=punchInfo.hasMetadata and \
                    punchInfo.isBAP1WT in [0,1] and \
                        (not punchInfo.isBlackListed)
                if isGood:
                    if punchInfo.isBAP1WT:
                        color='g'
                    else:
                        color='r'
                    plt.plot(rect[:,0],rect[:,1],'-',color=color,linewidth=3)
            plt.colorbar()
            plt.xticks([],[])
            plt.yticks([],[])
            plt.axis('on')
            plt.show()
