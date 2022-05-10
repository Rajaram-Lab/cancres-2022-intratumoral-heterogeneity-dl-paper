#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TCGA_PatchGeneration is used to generate all patches from TCGA used as the external 
testing set to evaluate the slide level models. 

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

sys.path.insert(0, ROOT_DIR)
import openslide as oSlide
import numpy as np

import pandas as pd
from glob import glob
import pickle

import Util.PatchGen as pg
import argparse
import multiprocessing as mp
import yaml

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
    parser = argparse.ArgumentParser(description='Automated job submission for patch generation on TCGA cohort.')
    parser.add_argument('--sampleStart', dest='sampleStart',
                        help='starting sample of parsed patch gen',
                        default=0, type=int)
    parser.add_argument('--sampleEnd', dest='sampleEnd',
                        help='end sample of parsed patch gen. Default is -1, which will process all files',
                        default=-1, type=int)
    args = parser.parse_args()
    return args

args=parse_args()
dataDir=projectPaths['DataDir']

# Set up Files
baseTCGADir=os.path.join(dataDir,projectPaths['Data']['ImageData'],'TCGA/')
allSvsFiles=[]
for subdir, dirs, files in os.walk(baseTCGADir,followlinks=True):
    for filename in files:
        filepath = subdir + os.sep + filename

        if filepath.endswith(".svs"):
            allSvsFiles.append(filepath)  

#%% Generate patches
    
def GeneratePatchesForSample(maskFiles,svsFileList,sampleNumber,patchDir):
    hdf5patchDataFile = os.path.join(patchDir, svsFileList[sampleNumber].rsplit('/',1)[1].replace('.svs','.hdf5'))
    slide = oSlide.open_slide(svsFileList[sampleNumber])
    
    
    classes={'background':0, 'blood':1, 'fat':2, 'normal':3, 'stroma':4, 'tumor':5}
    with open(maskFiles[sampleNumber],'rb') as handle:
      maskData=pickle.load(handle)
  
    mask=np.uint8(maskData[0]==classes['tumor'])
    maskToClassDict={1:'tumor'}

    
    patchSizeList=[224,224]
    downSampleLevelsList = [2,8]
    showProgress=False
    maxPatchesPerAnno=1200 # Maximum number of patches sampled from an annotation
    maxAvgPatchOverlap=8.0 # How tightly patches are allowed to overlap. 0 implies no overlap, 1 implies number of patches is selected so that combined area of patches= area of annotation
    minFracPatchInAnno=0.8 # What percentage of the patch must belong to same class as the center pixel, for the patch to be considered
    
    if not os.path.exists(hdf5patchDataFile):
        patchData, patchClasses, patchCenters = pg.PatchesFromMask( 
            slide = slide,
            mask = mask,
            downSampleFactors = downSampleLevelsList,
            patchSizes = patchSizeList,
            maskToClassDict = maskToClassDict,
            maxPatchesPerAnno = maxPatchesPerAnno,
            maxAvgPatchOverlap = maxAvgPatchOverlap,
            minFracPatchInAnno = minFracPatchInAnno,
            showProgress = showProgress)
        pg.SaveHdf5Data(
            hdf5Filename = hdf5patchDataFile, 
            patchData = patchData, 
            patchClasses = patchClasses, 
            patchCenters = patchCenters, 
            downSampleList = downSampleLevelsList, 
            patchSizeList = patchSizeList, 
            analyzedFile = svsFileList[sampleNumber])
        print(hdf5patchDataFile,' done!')
    else:
        print(hdf5patchDataFile,'\nAlready Exists!\n')
        

patchDir=os.path.join(dataDir,projectPaths['Data']['PatchData'],'TCGA/')
maskDir=os.path.join(dataDir,projectPaths['Tumor'],'TCGA/')
geneData=pd.read_csv(os.path.join(ROOT_DIR,projectPaths['Slide_Level']['Genetics']))

if not os.path.exists(patchDir):
    os.mkdir(patchDir)

fileMap={os.path.split(f)[-1]:f for f in glob(os.path.join(dataDir,projectPaths['Data']['ImageData'],'TCGA','*/*.svs'))}    
svsFileList=[fileMap[f] for f in geneData.svs.values]


maskFiles=[os.path.join(maskDir,os.path.split(file)[-1].replace('.svs','.pkl')) for file in svsFileList]

if args.sampleEnd<0:
   args.sampleEnd=len(maskFiles)
print ('Starting Pool')  
pool = mp.Pool(mp.cpu_count())
for sampleNumber in np.arange(args.sampleStart,args.sampleEnd):
  print('Submitting sample:', sampleNumber)
  pool.apply_async(GeneratePatchesForSample,args=(maskFiles,svsFileList,sampleNumber,patchDir))
pool.close()  
pool.join()   