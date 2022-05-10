#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to generate supplementary figure 9a-b (the color distributions
of samples across all cohorts). The process is wrapped into functions so that it 
can be called through the FigureMasterScript.

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

import sys
import yaml 
# import subprocess
# ROOT_DIR=subprocess.getoutput("git rev-parse --show-toplevel")
with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)

import openslide as oSlide


import Util.PatchGen as pg
import numpy as np
import h5py

import glob

import progressbar
import pickle
import pandas as pd


import matplotlib.pyplot as plt
import matplotlib.colors as colors
import Util.NormalizationCore as norm
from math import ceil
from skimage.transform import resize



dataDir=projectPaths['DataDir']

fontSize=figParams['fontsize']
mutColors=figParams['colors']
wtColors=figParams['wtColors']
geneList=figParams['geneList']
# %%

def LoadPatchData(hdf5FileList,classDict=None,returnSampleNumbers=False,
                  returnPatchCenters=False,scalesToUse=None):
    if len(hdf5FileList) == 0:
      print("Empty hdf5 list supplied. Exiting... \n")
      os._exit(1)
    # Find number of scales/patch dimensions
    numberOfPatches=0

    if classDict is None:
      uniqueLabels=set()
    else:
      uniqueLabels=np.array([k for k in classDict.keys()])
    
    patchSizes=[]
    for fileCounter,hdf5File in enumerate(hdf5FileList):
      with h5py.File(hdf5File, 'r') as f:
        if fileCounter==0:
          
          if scalesToUse is None:
              numberOfScales=f['patches'].attrs['numberOfScales']
              scalesToUse=np.arange(numberOfScales)
          else:   
              numberOfScales=len(scalesToUse)
          for s in scalesToUse:
            patchSizes.append(f['patches'+'/'+str(s)].attrs['patchSize'])
              
        classes=np.array(f['classes'][:],'U')
        if classDict is None:
          numberOfPatches=numberOfPatches+f['patches'].attrs['numberOfPatches']
          uniqueLabels.update(set(classes))
        else:
          isInDict= np.array([s in uniqueLabels for s in classes])
          numberOfPatches=numberOfPatches+np.sum(isInDict)
       
    if classDict is None:
      classDict={}
      for num,name in enumerate(np.sort(list(uniqueLabels))):
        classDict[name]=num

    #print(np.array(uniqueLabels))     
    patchData=[]
    #preAllocate patches across scales and classes 
    patchClasses=np.zeros((numberOfPatches))
    sampleNumbers=np.zeros((numberOfPatches))
    patchCenters=np.zeros((numberOfPatches,2))

    for scaleCounter,scale in enumerate(scalesToUse):
      print(numberOfPatches,patchSizes[scaleCounter],numberOfScales)
      patchData.append(np.zeros((numberOfPatches,patchSizes[scaleCounter],patchSizes[scaleCounter],3),dtype=np.uint8))
    #print('Initialized data in '+ str(time.time() - startTime)+' seconds')
    patchCounter=0  
    bar=progressbar.ProgressBar(max_value=len(hdf5FileList))
    for fileCounter,hdf5File in enumerate(hdf5FileList):
      
      hdf5Data=pg.LoadHdf5Data(hdf5File)
      #isInDict=np.in1d(hdf5Data['classLabels'],np.array(uniqueLabels,dtype='U'))
      isInDict= np.array([s in uniqueLabels for s in hdf5Data['classLabels']])
      nValidPatches=np.sum(isInDict)
      if(nValidPatches>0):
      #print(nValidPatches)
          try:
              patchClasses[np.uint32(np.arange(patchCounter,patchCounter+nValidPatches))]=np.array([classDict[c] for c in hdf5Data['classLabels'][isInDict]])
          except Exception as e:
              print(nValidPatches)
              print(np.arange(patchCounter,patchCounter+nValidPatches))
              print(e)
              sys.exit()
          sampleNumbers[np.uint32(np.arange(patchCounter,patchCounter+nValidPatches))]=fileCounter
          patchCenters[np.uint32(np.arange(patchCounter,patchCounter+nValidPatches)),:]=hdf5Data['patchCenters'][isInDict]
          for scaleCounter,scale in enumerate(scalesToUse):
            patchData[scaleCounter][np.uint32(np.arange(patchCounter,patchCounter+nValidPatches)),:,:,:]=hdf5Data['patchData'][scale]['patches'][isInDict]
            #print(hdf5Data['patchData'][scale]['patches'][isInDict])
      patchCounter=patchCounter+nValidPatches  
      bar.update(fileCounter)
    bar.finish()   
    if returnPatchCenters:
      return patchData, patchClasses,classDict,sampleNumbers,patchCenters
    elif returnSampleNumbers:
      return patchData, patchClasses,classDict,sampleNumbers
    else:
      return patchData, patchClasses,classDict 

    
#%% Generate slide RGB distribution data for TCGA and WSI
def GetSlideRGBData(): 
    targetDataSaveFile=os.path.join(ROOT_DIR, projectPaths['NormTarget'])
    # print(targetDataSaveFile)
    targetDataWSI=np.load(targetDataSaveFile)
    
    nPixelsToUse=100000
    outputPatchDir=os.path.join(dataDir,projectPaths['Data']['PatchData'],'TCGA/')
    
    hdf5List=glob.glob(os.path.join(outputPatchDir,'*.hdf5'))
    
    fullPatchData,_,_,fullSampleNumbers=LoadPatchData(hdf5List,returnSampleNumbers=True,scalesToUse=[0]) # Here it learns the classes from the data 
    
    fullPatchData=fullPatchData[0]
    
    inData=fullPatchData.reshape([-1,1,3])
    randIdx=np.random.choice(inData.shape[0],nPixelsToUse)
    targetDataTCGA=inData[randIdx,:,:]
    return targetDataWSI,targetDataTCGA

#%%Generate slide RGB distribution data for TMA data
def GetTmaRGBData():
    projectDict={'TMA1':[],'TMA2':[],'PDX1':[]}
    nPixelsToUse=100000
    visualizeMask=False
    
    for project in projectDict:
        assert project in ['TMA1','TMA2','PDX1']
        baseSvsDir=os.path.join(dataDir,projectPaths['Data']['ImageData'],
                                'TMA/')
            
        cohort=project 
        layoutDir=projectPaths['Region_Level']['TMALayout']
        layoutFile=os.path.join(layoutDir, cohort+'_Layout.csv')
        fullLayout=pd.read_csv(layoutFile).set_index('SVS')
        uniqueSvs=np.unique(fullLayout.index)
        svsList=[os.path.join(baseSvsDir,svs) for svs in uniqueSvs]
        
        startPixels=True
        
        tumorDir=os.path.join(dataDir,projectPaths['Tumor'],'TMA/')
        print('Selecting pixel data for', project)
        for svs in svsList:                  
            slide=oSlide.open_slide(svs)
            tumorPkl=tumorDir+svs.rsplit('/',1)[1].replace('svs','pkl')
            tissueMaskOG,_=pickle.load(open(tumorPkl ,'rb'))
            modelEffStrides=16
            padding = ceil(np.round(256/modelEffStrides))
            if len(tissueMaskOG)==2:
                sourceTumorMask=resize(np.pad(tissueMaskOG[0], [[0, padding],[0, padding]]),
                                       (slide.level_dimensions[2][1],slide.level_dimensions[2][0]))
            else:
                sourceTumorMask=resize(np.pad(tissueMaskOG, [[0, padding],[0, padding]]),
                                       (slide.level_dimensions[2][1],slide.level_dimensions[2][0]))
            
            if visualizeMask:
                slideImage=np.array(slide.read_region((0,0),2,slide.level_dimensions[2]))[:,:,range(3)]
                plt.figure(figsize=(10,10))
                plt.imshow(slideImage)
                plt.imshow(sourceTumorMask!=0,alpha=0.2,cmap=colors.ListedColormap(['w','b']))
                plt.title(os.path.split(svs)[-1])
                plt.show()
                
            sourcePixels=norm.SampleSlide(slide, sourceTumorMask!=0)
            if startPixels:
                combinedPixels=sourcePixels
                startPixels=False
            else:
                combinedPixels=np.concatenate([combinedPixels,sourcePixels], axis=0)
                
        projectDict[project]=combinedPixels[np.random.choice(combinedPixels.shape[0],nPixelsToUse),:,:]
    
    print(projectDict['TMA1'].shape,projectDict['TMA2'].shape,projectDict['PDX1'].shape)
    return projectDict
        
#%% Plot RGB histograms
def PlotColorDistribution():
    targetDataWSI,targetDataTCGA=GetSlideRGBData()
    projectDict=GetTmaRGBData()
    import seaborn as sns
    # from statsmodels.stats import weightstats as sms
    from scipy.stats import ks_2samp
    import skimage
    colorSpace='rgb'
    
    
    pixelDict={'WSI':targetDataWSI,
               'TCGA':targetDataTCGA,
               'TMA1':projectDict['TMA1'],
               'TMA2':projectDict['TMA2'],
               'PDX1':projectDict['PDX1']}
    
    
    # %
    
    lineColors=['midnightblue','royalblue',mutColors['BAP1'],'darkred','lightcoral']
    
    scores0,scores1,scores2={},{},{}
    plt.figure(figsize=(30,10))
    plt.subplot(131)
    for count,cohort in enumerate(pixelDict):
        if colorSpace=='rgb':
            sns.distplot(np.squeeze(pixelDict[cohort])[:,0],
                          hist=False,label=cohort,kde_kws=dict(linewidth=5),color=lineColors[count])
            plt.title('R',fontsize=16)
            ksScore=ks_2samp(np.squeeze(pixelDict['WSI'])[:,0],np.squeeze(pixelDict[cohort])[:,0])
        elif colorSpace=='hed':
            sns.distplot(skimage.color.rgb2hed(np.squeeze(pixelDict[cohort]))[:,0],
                         hist=False,label=cohort,kde_kws=dict(linewidth=5))
            plt.title('Hematoxylin',fontsize=16)
            ksScore=ks_2samp(skimage.color.rgb2hed(np.squeeze(pixelDict['WSI']))[:,0],skimage.color.rgb2hed(np.squeeze(pixelDict[cohort]))[:,0])
        scores0[cohort]=ksScore
    plt.legend(fontsize=16)
    plt.xlabel('Intensity Values',fontsize=16)
    plt.ylabel('Distribution Density',fontsize=16)
    
    plt.subplot(132)
    for count,cohort in enumerate(pixelDict):
        if colorSpace=='rgb':
            sns.distplot(np.squeeze(pixelDict[cohort])[:,1],
                          hist=False,label=cohort,kde_kws=dict(linewidth=5),color=lineColors[count])
            plt.title('G',fontsize=16)
            ksScore=ks_2samp(np.squeeze(pixelDict['WSI'])[:,1],np.squeeze(pixelDict[cohort])[:,1])
        elif colorSpace=='hed':
            sns.distplot(skimage.color.rgb2hed(np.squeeze(pixelDict[cohort]))[:,1],
                         hist=False,label=cohort,kde_kws=dict(linewidth=5))
            plt.title('Eosin',fontsize=16)
            ksScore=ks_2samp(skimage.color.rgb2hed(np.squeeze(pixelDict['WSI']))[:,1],skimage.color.rgb2hed(np.squeeze(pixelDict[cohort]))[:,1])
        scores1[cohort]=ksScore
    plt.legend(fontsize=16)
    plt.xlabel('Intensity Values',fontsize=16)
    plt.ylabel('Distribution Density',fontsize=16)
    
    
    plt.subplot(133)
    for count,cohort in enumerate(pixelDict):
        if colorSpace=='rgb':
            sns.distplot(np.squeeze(pixelDict[cohort])[:,2],
                          hist=False,label=cohort,kde_kws=dict(linewidth=5),color=lineColors[count])
            plt.title('B',fontsize=16)
            ksScore=ks_2samp(np.squeeze(pixelDict['WSI'])[:,2],np.squeeze(pixelDict[cohort])[:,2])
        elif colorSpace=='hed':
            sns.distplot(skimage.color.rgb2hed(np.squeeze(pixelDict[cohort]))[:,2],
                         hist=False,label=cohort,kde_kws=dict(linewidth=5))
            plt.title('DAB',fontsize=16)
            ksScore=ks_2samp(skimage.color.rgb2hed(np.squeeze(pixelDict['WSI']))[:,2],skimage.color.rgb2hed(np.squeeze(pixelDict[cohort]))[:,2])
        scores2[cohort]=ksScore
    plt.legend(fontsize=16)
    plt.xlabel('Intensity Values',fontsize=16)
    plt.ylabel('Distribution Density',fontsize=16)
