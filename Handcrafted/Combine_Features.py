"""
This file is used to combine all features across all WSI samples into a single
file. That single file will be used for all nuclear feature analysis performed 
in this project.

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

import yaml

with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
    
with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figureParams = yaml.full_load(file)


dataDir=projectPaths['DataDir']

import openslide as oSlide

import numpy as np


import pickle

import pandas as pd

from tqdm import tqdm

# %% Get matching image, mask and feature files

featuresDir=os.path.join(dataDir, projectPaths['Nuclear']['Features'])
sampleInfoFile=os.path.join(ROOT_DIR, projectPaths['Data']['AllSamples'])
svsDir=os.path.join(dataDir,projectPaths['Data']['ImageData'],'WSI/')
tumorMasksDir=os.path.join(dataDir,projectPaths['Tumor'],'WSI/')

allSamples = pd.read_csv(sampleInfoFile).drop(['Unnamed: 0'], axis=1)


featureFileList=[os.path.join(featuresDir, svs.replace('.svs','.pkl')) for svs in allSamples['svs'].to_list()]
featureFileList=[f for f in featureFileList if os.path.exists(f)]
tumorMaskList=[os.path.join(tumorMasksDir,os.path.split(f)[-1]) for f in featureFileList]
svsList=[os.path.split(f)[-1].replace('.pkl','.svs') for f in featureFileList]
numberOfFiles=len(featureFileList)      

# %% Calculate mean feature values for each slide

# Key Params
tumorStride=figureParams['TumorParams']['tumorStride']
minNuclearArea=figureParams['NuclearParams']['minNuclearArea']
maxNuclearArea=figureParams['NuclearParams']['maxNuclearArea']
numberOfFeatures=figureParams['NuclearParams']['numberOfFeatures']

# Final Results will be stored in a single matrix where each row
# is a single image, and each column is a single feature
combinedFeatMat=np.zeros((numberOfFiles,numberOfFeatures))

# Loop over all files
for fileNumber in tqdm(range(numberOfFiles)):
    
    # Get Region Mask and claculate tumor region
    maskData,regionLabels=pickle.load(open(tumorMaskList[fileNumber],'rb'))
    if type(maskData)==tuple:
        mask=maskData[0]
    else:
        mask=maskData
    isTumor=maskData==regionLabels.index('tumor')
    
    # Open slide to determine downsampling
    slide=oSlide.open_slide(os.path.join(svsDir,svsList[fileNumber]))
    downSampleFactor=slide.level_downsamples[1]*tumorStride
       
    # Load saved features    
    featureMat,featureNames=pickle.load(open(featureFileList[fileNumber],'rb'))
 
    # Find location of all nuclei to determine if they are in the tumor regions
    # Note: these need to be scaled down to the size of the tumor mask
    cX=np.minimum(np.uint32(np.round(featureMat[:,featureNames.index('Centroid_X')]/downSampleFactor)),isTumor.shape[1]-1)
    cY=np.minimum(np.uint32(np.round(featureMat[:,featureNames.index('Centroid_Y')]/downSampleFactor)),isTumor.shape[0]-1)
    
    # Get area of each nucleus (use for filtering out objects too large or small to be legitimate nuclei)
    area=featureMat[:,featureNames.index('Area')]
        
    #Preserve only nuclei in tumor area that are within a min and max area    
    isGood=np.logical_and(isTumor[cY,cX],
        np.logical_and(area>minNuclearArea,area<maxNuclearArea))
    
    # Drop features not related to size/shape/staining (e.g., index,position )
    goodFeatMat=featureMat[isGood,7:]
    goodFeatNames=featureNames[7:]
    
    # Represent Slide by median feature value across all nuclei
    avgFeatMat=np.nanmedian(goodFeatMat,axis=0)
    combinedFeatMat[fileNumber,:] =avgFeatMat   

allFeatNames=['Avg_'+n for n in featureNames[7:]]

# %%
saveFile=os.path.join(projectPaths['DataDir'], projectPaths['Nuclear']['CombinedFeats'])
pickle.dump([combinedFeatMat,svsList,allFeatNames],open(saveFile,'wb'))
