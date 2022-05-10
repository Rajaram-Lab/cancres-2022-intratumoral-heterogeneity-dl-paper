"""
WSI_PatchGeneration is used to generate patches for all WSI samples used in training 
the gene models and evaluating the slide gene models. Please note that this file 
does not discriminate between localized loss and universal loss when processing 
samples. Thus, universal loss samples will generate patches with only the loss label. 
The LocalizedLoss_PatchGeneration will generate patches correctly in a seperate directory.

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
with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)    
dataDir=projectPaths['DataDir']

import Util.PatchGen as pg
import openslide as oSlide
from glob import glob

import numpy as np

import pickle
import multiprocessing as mp
import argparse


# %% Read Start and End Samples from Command Line

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
    parser = argparse.ArgumentParser(description='Automated job submission to \
        generate patches on specified WSI samples.')
   
    parser.add_argument('--sampleStart', dest='sampleStart',
                        help='start index. Default is zero, which will start at the first sample',
                        default=0, type=int)
    parser.add_argument('--sampleEnd', dest='sampleEnd',
                        help='End index. Default is None, which will process all samples',
                        default=-1, type=int)

    args = parser.parse_args()
    return args

args=parse_args()
# %% Find list of annotated files for Tumor Level Classifier

svsDir=os.path.join(dataDir,projectPaths['Data']['ImageData'],'WSI/')
tumorDir=os.path.join(dataDir,projectPaths['Tumor'],'WSI/')

svsFileList=glob(svsDir+'*.svs')
pklFileList=[f.replace(svsDir,tumorDir).replace('svs','pkl') for f in svsFileList]



# %% Patch Generation 
    
def GeneratePatchesForSample(pklFileList,svsFileList,sampleNumber):
   
    outputPatchDir=os.path.join(dataDir,projectPaths['Data']['PatchData'],'WSI/')
    if not os.path.exists(outputPatchDir): 
            os.mkdir(outputPatchDir) 
    downSampleLevels=[1,4] # Downsampling factor relative to max (typically 20X). So 4 will give the 5X image. Adding multiple values gives patches at different scales
    patchSizeList=[224,224] # Patch size (we assume patches are square) in pixels. Specify pcdatch size separately for each scale in downSampleLevels
    showProgress=False
    maxPatchesPerAnno=1200 # Maximum number of patches sampled from an annotation
    maxAvgPatchOverlap=8.0 # How tightly patches are allowed to overlap. 0 implies no overlap, 1 implies number of patches is selected so that combined area of patches= area of annotation
    minFracPatchInAnno=0.8 # What percentage of the patch must belong to same class as the center pixel, for the patch to be considered
    
    pklFile=pklFileList[sampleNumber]   
    svsFile=svsFileList[sampleNumber]
    slide=oSlide.open_slide(svsFile)
    # hdf5File=os.path.join(outputPatchDir,os.path.split(pklFile)[-1].replace(version+'.pkl','.hdf5'))
    hdf5File=os.path.join(outputPatchDir,os.path.split(pklFile)[-1].replace('.pkl','.hdf5'))
    with open(pklFile,'rb') as handle:
        maskData=pickle.load(handle)
    
 
    mask=maskData[0]==maskData[1].index('tumor')

 
    maskToClassDict={1:'tumor'}
    if not os.path.exists(hdf5File):
        patchData,patchClasses,patchCenters=pg.PatchesFromMask(slide,mask,
                                                      downSampleLevels,patchSizeList,
                                                      maskToClassDict,
                                                      maxPatchesPerAnno=maxPatchesPerAnno,
                                                      showProgress=showProgress,
                                                      maxAvgPatchOverlap=maxAvgPatchOverlap,
                                                      minFracPatchInAnno=minFracPatchInAnno)

        pg.SaveHdf5Data(hdf5File,patchData,patchClasses,patchCenters,downSampleLevels,patchSizeList,svsFile)   
        os.chmod(hdf5File, 0o550)
        print(hdf5File,' done!')
    else:
        print(hdf5File,'Already Exists!')
        
# %%
if args.sampleEnd<0:
    args.sampleEnd=len(pklFileList)
print ('Starting Pool')  
pool = mp.Pool(mp.cpu_count())
for sampleNumber in np.arange(args.sampleStart,args.sampleEnd):
  print('Submitting sample:', sampleNumber)
  pool.apply_async(GeneratePatchesForSample,args=(pklFileList,svsFileList,sampleNumber))
pool.close()  
pool.join()   
