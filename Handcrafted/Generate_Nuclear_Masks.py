"""
This file is used to generate the nuclear masks that will be used for all nuclear
feature analysis.

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

import argparse 
import openslide as oSlide

import Util.UNet as UNet
import glob
# import re
import numpy as np

import pandas as pd
# import cv2 as cv2
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import yaml
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Model_Files.yaml')) as file:
    modelFiles  = yaml.full_load(file)
dataDir=projectPaths['DataDir']
# %%
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
                        help='start index',
                        default=0, type=int)
    parser.add_argument('--sampleEnd', dest='sampleEnd',
                        help='ending index',
                        default=-1, type=int)
    args = parser.parse_args()
    return args

args=parse_args()


sampleStart = args.sampleStart
sampleEnd = args.sampleEnd


# %%

modelSaveFile=os.path.join(dataDir, modelFiles['Nuclear_Model'])
model=UNet.LoadUNet(modelSaveFile,{'MeanIOU':UNet.MeanIOU})

# %%
allSampleFile=os.path.join(ROOT_DIR, projectPaths['Data']['AllSamples'])
allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)
negIdx=np.where(allSamples.BAP1_Positive.values==False)[0]
allNegSamples=allSamples.iloc[negIdx]

saveDir=os.path.join(dataDir,projectPaths['Nuclear']['Masks'])


svsDir=os.path.join(dataDir,projectPaths['Data']['ImageData'],'WSI/')

svsList=glob.glob(os.path.join(svsDir,'*.svs'))

for fileNumber in np.arange(sampleStart,sampleEnd):
    #svsFile=os.path.join(svsDir,svsList[fileNumber])
    svsFile=svsList[fileNumber]
    npFile=os.path.join(saveDir,os.path.split(svsFile)[1].replace('.svs','.npy'))
    if not os.path.exists(npFile):
    	try:
    		slide=oSlide.open_slide(svsFile)
    		
    		slideClasses=UNet.Profile_Slide_Fast(model,slide,borderSize=100)
    		nuclearMask=np.array(slideClasses==1,dtype=np.bool)
    		
    	    
    		np.save(npFile,nuclearMask)
    		print(npFile+' done!')
    	except:
    		
            	print('Error in '+ npFile+' skipping!')
    else:
        print(npFile+' exist,skipping!')
