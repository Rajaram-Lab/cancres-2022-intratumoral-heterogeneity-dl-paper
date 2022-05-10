"""
This file extracts all features from the nuclear masks that were generated for 
WSI cohort. The code will extract a total of 36 features for each nucleus and 
combine those features across all nuclei in each sample. It will save these features 
as npy files corresponding to each WSI sample.

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
# sys.path.insert(0, os.path.join(ROOT_DIR,'HandCrafted/'))
dataDir=projectPaths['DataDir']
import Handcrafted.FeatureExtractor as feat
from glob import glob
import openslide as oSlide
import numpy as np
import pickle
import time
import argparse 
# %% Parse command line args to get the range of files to analyze

def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Automated job submission')
    
    parser.add_argument('--startNum', dest='startNum',
                        help='Specify the starting number',
                        default=0, type=int)
    parser.add_argument('--stopNum', dest='stopNum',
                        help='Specify the stopping number',
                        default=-1, type=int)
    args = parser.parse_args()
    return args

args=parse_args()
startNum=args.startNum 
stopNum=args.stopNum
# %% Get matching svs files and nuclear masks

nuclearMasksDir=os.path.join(dataDir,projectPaths['Nuclear']['Masks'])
svsDir=os.path.join(dataDir,projectPaths['Data']['ImageData'],'WSI/')

nucMaskList=glob(os.path.join(nuclearMasksDir,'*.npy'))

svsFileList=[]
for f in nucMaskList:
   
    svsFile=os.path.join(svsDir,os.path.split(f)[-1].replace('.npy','.svs'))
    if os.path.exists(svsFile):
        svsFileList.append(svsFile)
    
# %% Run feature extraction
saveDir=os.path.join(dataDir,projectPaths['Nuclear']['Features'])

if(stopNum<0):
    stopNum=len(nucMaskList)
for fileNumber in range(startNum,stopNum):
    svsFile=svsFileList[fileNumber]    
    pklFile=os.path.join(saveDir,os.path.split(svsFile)[-1].replace('.svs','.pkl'))    
    if not os.path.exists(pklFile):
        
        nuclearMaskFile=nucMaskList[fileNumber]
        nuclearMask=np.load(nuclearMaskFile) # NOTE: this needs to be binary. So you need to drop the edge class
        
    
        slide=oSlide.open_slide(svsFile)
        
      
        hneImg=feat.ReadSlide(slide)
        print(svsFile+' loaded...')
        
        t0 = time.time()
        imgList=[hneImg]
        featList=[feat.Location(),feat.Size(),feat.Shape(),feat.Convexity(),
                  feat.IntensityStats(0,'H_Int',transform=feat.GetH),
                  feat.IntensityStats(0,'LabL_Int',transform=feat.GetLabL),
                  feat.Haralick_Texture(0, -1, 1, 'Hint',transform=feat.GetH)]
        featMat,featNames=feat.ExtractFeatures(nuclearMask, featList,imgList=imgList)
        t1 = time.time()    
        print(t1-t0)   
        
    
        pickle.dump((featMat,featNames),open(pklFile,'wb'))
        print(svsFile+' done!')
    else:
        print(pklFile+' exists, skipping!')