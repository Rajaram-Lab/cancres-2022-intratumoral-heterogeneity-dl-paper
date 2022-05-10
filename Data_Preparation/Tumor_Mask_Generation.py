"""
This script generates tumor masks for each of the samples in the specified cohort.

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


import Util.DLUtils as dlutil

import openslide as oSlide
import numpy as np
import pickle
from glob import glob 

import yaml 
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file) 
with open(os.path.join(ROOT_DIR,'Parameters/Model_Files.yaml')) as file:
    modelFiles = yaml.full_load(file) 
with open(os.path.join(ROOT_DIR,'Data_Files/TMA/Cohort_Files.yaml')) as file:
    tmaFiles = yaml.full_load(file)
    
dataDir=projectPaths['DataDir']

import argparse
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
    parser = argparse.ArgumentParser(description='Automated job submission for tumor \
        masks generation for specified cohort.')
    
    parser.add_argument('--sampleStart', dest='sampleStart',
                        help='Which sample to start on for the cohort'+\
                            "Accepted values must be within size of the cohort",
                        default=0, type=int)
    parser.add_argument('--sampleEnd', dest='sampleEnd',
                        help='Last sample to process for the cohort'+\
                            "Accepted values must be within size of the cohort",
                        default=None, type=int)
    parser.add_argument('--singleCohort', dest='singleCohort',
                        help='Specify whether you want to generate masks for a single cohort.'+\
                            'By default, this will generate masks for all data cohorts',
                        default=None, type=str)
    args = parser.parse_args()
    return args

args=parse_args()
#%%
dataSets=['WSI','TCGA','TMA1','TMA2','PDX1']
classifierFile=os.path.join(dataDir,modelFiles['Tumor_Model'])

#%%
def GenerateTumorMasks(svsFile, pklFile,classifierFile):
    tumorClassifier=dlutil.Classifier()
    tumorClassifier.Load(classifierFile)
    slide=oSlide.open_slide(svsFile)
    tumorClasses=dlutil.Profile_Tumor(tumorClassifier,slide,returnScores=False)
    pickle.dump( [tumorClasses,tumorClassifier.labelNames], open( pklFile, 'wb' ), protocol=4 )
    
def GenerateTumorMasksFast(svsFile,pklFile,classifierFile,boxSize,batchSize,cohort):
    myClassifier = dlutil.Classifier() 
    myClassifier.Load(classifierFile) 
    tumorModel=myClassifier.model
    
    slide=oSlide.open_slide(svsFile)

    slideMpp = np.mean([float(slide.properties[p]) for p in slide.properties if 'mpp' in p.lower()]) 
    downSampleMultiplier = int(np.round(0.5 / slideMpp)) 
    dsf = np.power(4, myClassifier.magLevel) * downSampleMultiplier
    tumorMaskData=dlutil.Profile_Slide_Fast(
                            tumorModel,slide,
                            myClassifier.effectiveStride,
                            myClassifier.patchSize,
                            myClassifier.numberOfClasses,
                            downSampleFactor=dsf, #4 before
                            useMultiprocessing=False,
                            batchSize=batchSize,
                            boxHeight=boxSize, boxWidth=boxSize)
    
    if cohort=="TMA2" or cohort == "TCGA":
        pickle.dump(tumorMaskData,open(pklFile,'wb'))
        
    else:
        sourceMaskLabels=myClassifier.labelNames
        tumorMaskHardCalls=tumorMaskData[0]
        pickle.dump((tumorMaskHardCalls,sourceMaskLabels),open(pklFile,'wb'))
        
#%% Generate masks for each of the samples in the directory

if args.singleCohort is None:
    dataSetsToProfile = dataSets

elif args.singleCohort is not None and args.singleCohort in dataSets:
    dataSetsToProfile = [args.singleCohort]
    
elif args.singleCohort is not None and args.singleCohort not in dataSets:
    raise ValueError("Incorrect cohort value supplied")
    
# The following section will loop through all available datasets and 
for cohort in dataSetsToProfile:
    imageRoot=os.path.join(dataDir,projectPaths['Data']['ImageData'])
    maskRoot=os.path.join(dataDir,projectPaths['Tumor'])
    
    if cohort =='WSI': # This branch will select all WSI svs files
        maskDestination=os.path.join(maskRoot, 'WSI/')
        baseWSIDir=os.path.join(imageRoot,'WSI/')
        svsFileList = glob(os.path.join(baseWSIDir,'*.svs'))
                    
        pklFileList=[f.replace('svs','pkl').replace(baseWSIDir,maskDestination) \
                     for f in svsFileList]
            
    elif cohort == 'TCGA': # This branch wills select all TCGA svs files
        # Set up TCGA Files
        baseTCGADir=os.path.join(imageRoot,'TCGA/')
        maskDestination=os.path.join(maskRoot, 'TCGA/')
        svsFileList=[]
        for subdir, dirs, files in os.walk(baseTCGADir,followlinks=True):
            for filename in files:
                filepath = subdir + os.sep + filename
        
                if filepath.endswith(".svs"):
                    svsFileList.append(filepath)  
        
        pklFileList=[os.path.join(maskDestination,f.rsplit('/',1)[1].replace('svs','pkl')) \
                     for f in svsFileList]
            
    else: # This branch will select the svs files for the corresponding TMA cohort
        maskDestination=os.path.join(maskRoot, 'TMA/')
        baseTMADir=os.path.join(imageRoot, "TMA/")
        svsFileNames=tmaFiles[cohort]
        svsFileList = [os.path.join(baseTMADir,f) for f in svsFileNames]
                    
        pklFileList=[f.replace('svs','pkl').replace(baseTMADir,maskDestination) \
                     for f in svsFileList]
        
    svsFileListToProcess = svsFileList[args.sampleStart:args.sampleEnd]
    pklFileListToProcess = pklFileList[args.sampleStart:args.sampleEnd]
    
    if len(svsFileListToProcess) == 0:
        raise Exception("Please check sampleStart and sampleEnd arguments supplied")

    
    if cohort in tmaFiles:
        boxSize=1024
        batchSize=1
    else:
        boxSize=1000
        batchSize=2
    
    for counter,svsFile in enumerate(svsFileListToProcess):
        pklFile=pklFileListToProcess[counter]
        if not os.path.exists(pklFile):     
            if cohort=='WSI':
                GenerateTumorMasks(svsFile, pklFile,classifierFile)
            else:
                GenerateTumorMasksFast(svsFile, pklFile,classifierFile,
                                       boxSize, batchSize, cohort)
            print(svsFile+ ' Done!'  )
        else:
            print(pklFile+" Already Exists!")
        