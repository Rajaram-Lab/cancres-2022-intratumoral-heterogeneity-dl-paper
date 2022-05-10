#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to generate supplementary figure 7 (focal sample image plots). 
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


import openslide as oSlide
#%matplotlib inline
import Util.PatchGen as pg 
import matplotlib.pyplot as plt

import numpy as np
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from skimage.transform import resize

import pickle
import copy
from matplotlib import cm
from matplotlib.colors import ListedColormap
import Util.plotUtils as PlUtils

import yaml

with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)   
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
dataDir=projectPaths['DataDir']


dataDir=projectPaths['DataDir']

fontSize=figParams['fontsize']
mutColors=figParams['colors']
wtColors=figParams['wtColors']
geneList=figParams['geneList']
geneScores={'BAP1':[],'PBRM1':[],'SETD2':[]}
geneLabels={'BAP1':[],'PBRM1':[],'SETD2':[]}
#%%
def DefineColorMap(gene):
    mutCap=mutColors[gene]
    wtCap=wtColors[gene]
    
    newcolors=np.ones((256,4))
    # viridis = cm.get_cmap('Blues', 256)
    for i in range(len(mutCap)):
        if mutCap[i]==0:
            newMutArray=np.linspace(0,1,128)
        else: 
            newMutArray=np.linspace(mutCap[i],1,128)
        if wtCap[i]==0:
            newWtArray=np.linspace(0,1,128)
        else: 
            newWtArray=np.linspace(wtCap[i],1,128)
        combinedColorArray=np.concatenate((newMutArray,np.flip(newWtArray)))
        newcolors[:,i]=combinedColorArray
    newCmap=ListedColormap(newcolors,'GreedRed')
    
def PlotGridFocalMaps(focalFile,xmlFile,svsFile):
    
    mpp=0.4936  #microns per pixel
    mm=1000/mpp  #microns to millimeter
    downSampleFactor=100  # dsf=stride*dsf = 100 * 1
    dimension=np.int32(np.round(mm/downSampleFactor)) 
    threshold=0.7   #threshold /project/bioinformatics/Rajaram_lab/shared/Driver_Mutations_From_HnE/Sample_Results/Segmentation/Whole_Slide/SingleOutput_NewCV/VGG19_BAP1/19366_avg.pklfor the ratio of tumor composition that must be present to use grid square

    slide=oSlide.open_slide(svsFile)
    slideImage=np.array(slide.read_region((0,0),1,slide.level_dimensions[1]))[:,:,range(3)]

    svsDir=os.path.join(dataDir,projectPaths['Data']['ImageData'],'WSI/')
    tumorDir=os.path.join(dataDir, projectPaths['Tumor'],'WSI/')
    tumorPklFile=svsFile.replace(svsDir,tumorDir).replace('svs','pkl')
    
    with open(tumorPklFile,'rb') as handle:
        maskData=pickle.load(handle)
 
    tumorMask=np.uint8(maskData[0]==maskData[1].index('tumor')) 
        
    with open(focalFile,'rb') as handle:
        focalMaskData=pickle.load(handle)

        
    focalActMask=focalMaskData[1][:,:,1]
    focalMask=focalMaskData[0]
    annoMask, maskDict= pg.MaskFromXML(xmlFile,'Focal_BAP1',slideDim=(slide.level_dimensions[0][0],slide.level_dimensions[0][1]),
                                          downSampleFactor=downSampleFactor,distinguishAnnosInClass=False)
    negMask=PlUtils.ResizeOneHotEncode(annoMask,classes=2, dimensions=focalMask.shape)
    
    tumorMask=np.float16(PlUtils.ResizeOneHotEncode(tumorMask, classes=2, dimensions=focalMask.shape))
    tmp=copy.copy(tumorMask)
    tmp[tmp==0]=-1
    finalMask=negMask+tmp
    finalMask[finalMask==-1]=0
    groundTruthMask=3-finalMask
    groundTruthMask[groundTruthMask==3]=np.nan
    x=np.arange(0,focalMask.shape[0],dimension)
    y=np.arange(0,focalMask.shape[1],dimension)
    
    gtMask,fcMask,faMask = groundTruthMask.copy(),focalMask.copy(),focalActMask.copy()
    blMask=np.ones(gtMask.shape)
    x1=0
    for row in x:
        y1=0
        for col in y:
            gridZone=tumorMask[x1:row+dimension,y1:col+dimension]
            if np.min(gridZone.shape) == dimension:
                if np.mean(gridZone)>threshold:
                    predictionZone=focalActMask[x1:row+dimension,y1:col+dimension]
                    callZone=focalMask[x1:row+dimension,y1:col+dimension]
                    
                    predictedAvg=np.mean(predictionZone)
                    callAvg=np.round(np.mean(callZone))
                    gtAvg=np.round(np.nanmean(groundTruthMask[x1:row+dimension,y1:col+dimension]))
                    
                    gtMask[x1:row+dimension,y1:col+dimension]=gtAvg
                    fcMask[x1:row+dimension,y1:col+dimension]=callAvg
                    faMask[x1:row+dimension,y1:col+dimension]=predictedAvg
                    
                    blMask[x1:row+dimension,y1:col+dimension]=0
            y1=y1+dimension
        x1=x1+dimension
    
    faMask2=copy.copy(faMask)
    faMask2[blMask==1]=np.nan
    slideImg=np.int64(resize(slideImage,fcMask.shape)*255)
    fcMask=fcMask/255
    fcMask[blMask==1]=np.nan
    gtMask=gtMask/255
    gtMask[blMask==1]=np.nan

    bottom = cm.get_cmap('Greens', 128)
    top = cm.get_cmap('Reds', 128)
    newcolors = np.vstack((top(np.array(list(reversed(np.linspace(0,1,128))))),
                            bottom(np.linspace(0, 1, 128))))
    newCmap = ListedColormap(newcolors, name='GreenRed')
    
    plt.figure(figsize=(30,15))
    plt.subplot(161)
    ax = plt.gca()
    ax.imshow(slideImg)
    plt.title('Slide Image',fontsize=fontSize)
    plt.axis('off')
    
    plt.subplot(162)
    ax1 = plt.gca()
    ax1.imshow(focalActMask,cmap=newCmap,vmin=0,vmax=1)
    plt.title("Activation Map",fontsize=fontSize)
    ax1.axis('off')

    
    plt.subplot(163)
    ax2 = plt.gca()

    ax2.imshow(faMask2,cmap=newCmap,vmin=0,vmax=1)
    plt.title("Region Activation Map",fontsize=fontSize)
    ax2.axis('off')
    
    
    plt.subplot(164)
    ax3 = plt.gca()
    ax3.axis('off')
    ax3.imshow(slideImg,alpha=1)
    ax3.imshow(fcMask,alpha=0.85,cmap=newCmap)
    plt.title("Classifier Prediction Map",fontsize=fontSize)
    ax3.axis('off')
    
    plt.subplot(165)
    ax4 = plt.gca()
    ax4.imshow(slideImg,alpha=1)
    ax4.imshow(gtMask,alpha=0.85,cmap=newCmap)
    plt.title("Ground Truth Map",fontsize=fontSize)
    ax4.axis('off')
    
    return slideImg,focalActMask,faMask2,fcMask,gtMask
   
#%% Define focal holdout cases
def DefineFocalData():
    geneName='BAP1'    
    focalSamples=PlUtils.Load_Test_Samples('focal',geneName)
    
    
    svsDir=os.path.join(dataDir,projectPaths['Data']['ImageData'],'WSI/')
    xmlDir=os.path.join(ROOT_DIR, projectPaths['Region_Level']['XmlDir'])
    activationDir=os.path.join(dataDir, projectPaths['ResponseData'],
                               'Region_Level/WSI/')
    maskDir=activationDir+geneName+'/'
    
    xmlFileList=[xmlDir+sample.replace('svs','xml') for sample in focalSamples]
    
    temp=[x.replace('xml','svs') for x in xmlFileList]
    svsFileList=[x.replace(xmlDir, svsDir) for x in temp]
    
    tempList=[x.replace(xmlDir,maskDir) for x in xmlFileList]
    
    focalActivationList=[x.replace('.xml','_avg.pkl') for x in tempList]
    
    return focalActivationList,xmlFileList, svsFileList

#%% Supp figure 7. Plot all holdouts. 
    
def SuppFigure7Plots():
    focalActivationList,xmlFileList, svsFileList=DefineFocalData()
    
    for i in range(len(xmlFileList)):
        slideImg,focalActMask,faMask2,fcMask,gtMask=PlotGridFocalMaps(focalActivationList[i],xmlFileList[i],svsFileList[i]) 

#%% Figure 3: Plot focal case with grid lines

def Figure4Plots():
    import matplotlib.ticker as plticker
    
    focalActivationList,xmlFileList, svsFileList=DefineFocalData()
    mpp=0.4936  #microns per pixel
    mm=1000/mpp  #microns to millimeter
    downSampleFactor=100  # dsf=stride*dsf =
    dimension=np.int32(np.round(mm/downSampleFactor)) 
    myInterval=dimension
    
    i=2  # this is the image we used
    plotReturns=PlotGridFocalMaps(focalActivationList[i],xmlFileList[i],svsFileList[i]) 
    slideImg,focalActMask,faMask2,fcMask,gtMask=plotReturns
    
    bottom = cm.get_cmap('Greens', 128)
    top = cm.get_cmap('Reds', 128)
    newcolors = np.vstack((top(np.array(list(reversed(np.linspace(0,1,128))))),
                            bottom(np.linspace(0, 1, 128))))
    newCmap = ListedColormap(newcolors, name='GreenRed')
    titles=['Classifier Hard Call','Ground Truth']
    
    # 4a slide image
    fig=plt.figure(figsize=(10,15))
    # crop and rotate
    plt.imshow(slideImg)
    
    # 4b Focal Activation
    fig=plt.figure(figsize=(10,15))
    plt.imshow(focalActMask,cmap=newCmap,vmin=0,vmax=1)
    
    # 4c Focal average
    fig=plt.figure(figsize=(10,15))
    ax2=fig.add_subplot(111)
    
    # Remove whitespace from around the image
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # Set the gridding interval: here we use the major tick interval
    # loc = plticker.MultipleLocator(myInterval)
    
    ax2.xaxis.set_major_locator(plticker.MultipleLocator(myInterval))
    ax2.yaxis.set_major_locator(plticker.MultipleLocator(myInterval))
    
    # Add the grid
    ax2.grid(which='major', axis='both', linestyle='-',color='gray',linewidth=1,b=True)
    # Add the activation image
    ax2.imshow(faMask2,alpha=0.9,cmap=newCmap,vmin=0,vmax=1)
    plt.title('Activation')
    
    # 4d and 4e
    for count,image in enumerate([fcMask,gtMask]):
        # Set up figure
        fig=plt.figure(figsize=(10,15))
        ax=fig.add_subplot(111)
        # Remove whitespace from around the image
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
        # Set the gridding interval: here we use the major tick interval
        # loc = plticker.MultipleLocator(myInterval)
        
        ax.xaxis.set_major_locator(plticker.MultipleLocator(myInterval))
        ax.yaxis.set_major_locator(plticker.MultipleLocator(myInterval))
        
        
        # Add the slide image
        ax.imshow(slideImg,alpha=1)
        # Add the grid
        ax.grid(which='major', axis='both', linestyle='-',color='gray',linewidth=1,b=True)
        # Add the superimposed image
        ax.imshow(image,alpha=0.9,cmap=newCmap)
        plt.title(titles[count])
    
