#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tessellated_Inference is used to evaluate all region gene models across all testing cohorts.
The required model parameters and model file names are found in the parameter files :
(Training_Params.yaml and Model_Files.yaml).

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
sys.path.insert(0,ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR,'External/tiler/'))
CUDA_VISIBLE_DEVICES=0

import yaml

import openslide as oSlide
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from math import ceil
from skimage.transform import resize
import Util.NormalizationCore as norm
import argparse
import pickle
import pandas as pd 

import tiler


with open(os.path.join(ROOT_DIR, 'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Training_Params.yaml')) as file:
    trainParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Data_Files/TMA/Cohort_Files.yaml')) as file:
    tmaFiles = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Model_Files.yaml')) as file:
    modelFiles = yaml.full_load(file)
dataDir=projectPaths['DataDir']
#%%
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
    parser = argparse.ArgumentParser(description='Automated job submission for Tessellated inference \
        of specified cohort')
    parser.add_argument('--cohort',dest='cohort',
                        help='WSI,TMA1,TMA2,PDX1',
                        default='PDX1',type=str)
    parser.add_argument('--foldsToProfile', dest='foldsToProfile',
                        help='specify fold to use for tessellation evaluation',
                        nargs='+',default=[0,1,2], type=int)
    parser.add_argument('--gene',dest='gene',
                        help='BAP1,PBRM1,SETD2',
                        default='BAP1',type=str)
    parser.add_argument('--performNorm',dest='performNorm',
                        help='0-False, 1-True',
                        default=0,type=int)
    args = parser.parse_args()
    return args

args=parse_args()
cohort=args.cohort
performNorm=args.performNorm

# Establish path to all response data 
responseData=os.path.join(dataDir,projectPaths['ResponseData'])

# WSI activation diretory according to the gene indicated in arg parser
activationDirWSI=os.path.join(responseData,'Region_Level','WSI',args.gene+'/')

# Normalization results will be saved in a different folder than no norm results
if performNorm:
    activationDirTMA=os.path.join(responseData,'Region_Level','TMA_Norm/')
else:
    activationDirTMA=os.path.join(responseData,'Region_Level','TMA_NoNorm/')
    
# If either of the activation directories do not exist, create the directories    
if not os.path.exists(activationDirWSI):
    os.mkdir(activationDirWSI)
    
if not os.path.exists(activationDirTMA):
    os.mkdir(activationDirTMA)
#%% Functions

def Load_Tessellation_Model(gene,fold):
    modelType='FCN_Models'
    modelDir=os.path.join(dataDir,modelFiles[modelType]['BaseDir'])
     
    modelList=modelFiles[modelType][gene]
    modelFile = os.path.join(modelDir,modelList[fold])
    tempModel=load_model(modelFile)
    model=Model(inputs=tempModel.input,outputs=tempModel.get_layer(index=-2).output)
    return model

def Crop_Tessellation(outMat,slide,slideGen,dsf):
    downSampleFactor=dsf
    (slideWidth,slideHeight)=slide.dimensions
    outHeight=int(np.floor(slideHeight/(downSampleFactor*stride)))
    outWidth=int(np.floor(slideWidth/(downSampleFactor*stride)))
    
    outMatCropped=outMat[:outHeight,:outWidth,:]   
    return outMatCropped

def Reconstruct_Slide(res,slideGen):

    nPatch=res.shape[0]
    numberOfChannels=res.shape[-1]
    resFlat=np.squeeze(res)
    idxInBatch=np.remainder(np.arange(nPatch),slideGen.batch_size)
    batchNumber=np.uint32(np.floor(np.arange(nPatch)/slideGen.batch_size))
    
    # offset based on location within batch
    offC,offR=np.meshgrid(np.arange(nBx),np.arange(nBx))
    offR=offR.flatten()
    offC=offC.flatten()    
   
    
    nBatch=len(slideGen)
    # Location of batch 
    nGridR=len(np.unique(slideGen.rVals))
    nGridC=len(np.unique(slideGen.cVals))
    rVals,cVals=np.meshgrid(np.arange(0,nGridR),np.arange(0,nGridC))  
   
    rVals=rVals.flatten()*nBx
    cVals=cVals.flatten()*nBx   
    
    assert len(rVals)==nBatch
    
    outMat=np.zeros((nGridR*nBx,nGridC*nBx,numberOfChannels))

    
    idR=rVals[batchNumber]+offR[idxInBatch]
    idC=cVals[batchNumber]+offC[idxInBatch]
    
    for c in range(numberOfChannels):
        temp=np.zeros((nGridR*nBx,nGridC*nBx))
        
        temp[idR,idC]=resFlat[:,c]
        outMat[:,:,c]=temp        
    return outMat

def Check_File_Exists(file):
    exists=os.path.exists(file)
    if exists:
        print(file, ' Already exists. Skipping!')
    return exists

def Save_Tessellation(activationMap, filePath):
    pickle.dump(activationMap,open(filePath,'wb'))
    os.chmod(pklFile, 0o550)
    
# %% Custom Generator


class SlidePatchGenerator(Sequence):
  """ Uses Keras' generator to create boxes out of the slide. """
  def __init__(self,slide,patchSize=224,stride=100,nBx=16,
                shuffle=False, downSampleFactor=1, 
                preproc_fn=lambda x:np.float32(x)/255):
      """ svsDir
      Initialize the generator to create boxes of the slide. 
      
      ### Returns
      - None
      
      ### Parameters:
      - `slide: slide`  The loaded slide object.
      - `patchSize: int`  Height/width of the output patches
      - `stride: int`  pixels between successive patches.
      - `nBx: int`  batch size will be square of this number
      
      """
      
      self.slide=slide
      self.dsf=downSampleFactor
      self.patchSize=patchSize
      
      
      self.boxHeight=int(nBx*stride+(patchSize-stride))
      self.boxWidth=int(nBx*stride+(patchSize-stride))
      self.batch_size = int(nBx*nBx)
      self.shuffle = shuffle     
      
      (self.slideWidth,self.slideHeight)=slide.dimensions
      self.rVals,self.cVals=np.meshgrid(np.arange(0,self.slideHeight,self.dsf*nBx*stride),
                                        np.arange(0,self.slideWidth,self.dsf*nBx*stride))          
      self.numberOfBoxes=self.rVals.size
      self.rVals.resize(self.rVals.size)
      self.cVals.resize(self.cVals.size)
      
      self.preproc=preproc_fn
      self.patchify = tiler.Tiler(data_shape=(self.boxHeight,self.boxWidth,3),
                    tile_shape=(patchSize, patchSize,3),
                    channel_dimension=2,overlap=patchSize-stride)
  
  def __len__(self):
      """
      Denotes the number of batches per epoch
      
      ### Returns
      - `int`  Number of batches for the keras generator.
      
      ### Parameters
      - None
      
      """
      return self.numberOfBoxes
  
  def __getitem__(self, index):
      """
      Generate one batch of data
      
      ### Returns
      - `X: np.array()` of shape (numBoxesInBatch, numRows, numCols, numChannels)
      - `Y: List(np.array(), np.array())` where first numpy array is row values in the batch and second numpy array is col value in the batch.
      
      ### Parameters
      - `index: int`  batchIndex to be called.
      """
      # Generate indexes of the batch
      
      Y=[0,0]
      assert self.dsf in [1,2], "Slide must be 20X or 40X"
      
      r=self.rVals[index]
      c=self.cVals[index]
      
      if self.dsf==1:  # -- Enter this branch if the slide is 20X     
          X=np.zeros((self.batch_size,self.patchSize,self.patchSize,3),dtype=np.float32)   
          
          img=np.zeros((self.boxHeight,self.boxWidth,3))
          
          imgHeight=int(np.minimum(self.boxHeight,self.slideHeight-(r)))
          imgWidth=int(np.minimum(self.boxWidth,self.slideWidth-(c)))
          
          img[0:imgHeight,0:imgWidth]=np.array(self.slide.read_region((c,r),0,(imgWidth,imgHeight)))[:,:,range(3)]
         
      
      elif self.dsf==2:  #-- Enter this branch if the slide is 40X
          
          self.magLevel=0
        
          # same operations as the other branch branch except multiple by dsf
          boxHeightRaw=int(self.dsf*self.boxHeight)
          boxWidthRaw=int(self.dsf*self.boxWidth)
        
          imgHeight=int(np.minimum(boxHeightRaw,self.slideHeight-(r)))
          imgWidth=int(np.minimum(boxWidthRaw,self.slideWidth-(c)))
        
          img=np.zeros((boxHeightRaw,boxWidthRaw,3))
          
          img[0:imgHeight,0:imgWidth]=np.array(self.slide.read_region((c,r),self.magLevel,(imgWidth,imgHeight)))[:,:,range(3)]
                    
          img=resize(img,(self.boxHeight,self.boxWidth))
          
      patches=[batch for _, batch in self.patchify(img, batch_size=self.batch_size)][0]
      X=self.preproc(patches)
      
      return X, Y
  
  def on_epoch_end(self):
       'Updates indexes after each epoch'
       # self.indexes = np.arange(self.numberOfPatches)

def Perform_Norm(svs,slide):
    
   
    targetDataSaveFile=os.path.join(ROOT_DIR, projectPaths['NormTarget'])
    targetData=np.load(targetDataSaveFile)
    percentileCutoff=95
    normalizationScheme='vahadane'#'macenko'#'vahadane'#
    modelEffStrides=16   # tumor model, used for padding tumor mask
    modelEffKernelSize=256  # tumor model, used for padding tumor mask
    
    tumorDir=os.path.join(dataDir,projectPaths['Tumor'],'TMA/')
    tumorPkl=tumorDir+svs.rsplit('/',1)[1].replace('.svs','.pkl')
    tumorMaskData,_=pickle.load(open(tumorPkl ,'rb'))
    padding = ceil(np.round(modelEffKernelSize/modelEffStrides))      
    sourceTumorMask=resize(np.pad(tumorMaskData, [[0, padding],[0, padding]]),
                           (slide.level_dimensions[2][1],slide.level_dimensions[2][0]))
    
                          
    myNormalizer=norm.StainNormalizer(normalizationScheme)
    myNormalizer.fit(np.uint8(targetData),None,slide,\
                      sourceTumorMask!=0,percentileCutoff=percentileCutoff)
        
    return myNormalizer.transformFull
    
    
#%% Perform profiling using tessellation 
nBx=trainParams['Tessellation']['nBox'] # 16 by default. Results in 256 batch size 
patchSize=trainParams['Tessellation']['patchSize']  #224 by default
stride=trainParams['Tessellation']['stride']  # 100 default, 32 to simulate FCN model

if cohort != 'WSI':  # -- Enter this branch for TMA inference
    assert cohort in ['TMA1','PDX1','TMA2']
    filesToUse=tmaFiles[cohort]
    baseSvsDir=os.path.join(dataDir,projectPaths['Data']['ImageData'],"TMA/")
    svsList=[os.path.join(baseSvsDir,svs) for svs in filesToUse]
    dsf=trainParams['DSF']['TMA'][cohort]
    saveDir=activationDirTMA
        
    svsToPkl=lambda f: os.path.join(saveDir,os.path.split(f)[-1].replace('.svs','_Fold-'+str(foldNum)+'.pkl')) 

    for foldNum in args.foldsToProfile:
            
            model=Load_Tessellation_Model(args.gene, foldNum)
            
            for svs in svsList:
                pklFile=svsToPkl(svs)
                saveFile=os.path.join(saveDir,pklFile)
                if not Check_File_Exists(saveFile):
                    slide=oSlide.open_slide(svs)
                   
                    if performNorm:
                        preproc_fn=Perform_Norm(svs, slide)
                        
                        slideGen=SlidePatchGenerator(slide,patchSize=patchSize,downSampleFactor=dsf,
                                                     nBx=nBx,stride=stride,preproc_fn=preproc_fn)
                    else: 
                        slideGen=SlidePatchGenerator(slide,patchSize=patchSize,downSampleFactor=dsf,
                                                     nBx=nBx,stride=stride)
                        
                    res=model.predict(slideGen,verbose=1)
                    actMap=Reconstruct_Slide(res, slideGen)
                    cropActMap=Crop_Tessellation(actMap, slide, slideGen, dsf)
                    
                    Save_Tessellation(cropActMap, saveFile)

elif cohort == 'WSI':
    baseSvsDir=os.path.join(dataDir, projectPaths['Data']['ImageData'],'WSI/')
    allSampleFile=os.path.join(ROOT_DIR,projectPaths['Data']['AllSamples'])
    foldsIdx=os.path.join(ROOT_DIR,projectPaths['Data']['FoldsIdx'])
    saveDir=activationDirWSI
    gene=args.gene
    dsf=1
    

    allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)
    with open(foldsIdx, 'rb') as f:
        folds = pickle.load(f)
    validIdx=folds[3]
    testSamples=allSamples.iloc[validIdx]
    
    # Create svs list 
    svsList=[os.path.join(baseSvsDir,svs) for svs in testSamples.svs.values]
    
    
    for foldNum in args.foldsToProfile:
        svsToPkl=lambda f: os.path.join(saveDir,os.path.split(f)[-1].replace('.svs','_Fold-'+str(foldNum)+'.pkl')) 
        model=Load_Tessellation_Model(args.gene, foldNum)
        for svs in svsList:
            pklFile=svsToPkl(svs)
            saveFile=os.path.join(saveDir,pklFile)
            if not Check_File_Exists(saveFile):
                
                slide=oSlide.open_slide(svs)
                slideGen=SlidePatchGenerator(slide,patchSize=patchSize,nBx=nBx,
                                             stride=stride)
                                
                res=model.predict(slideGen,verbose=1)
                actMap=Reconstruct_Slide(res, slideGen)
                cropActMap=Crop_Tessellation(actMap, slide, slideGen, dsf)
                
                Save_Tessellation(cropActMap,saveFile)
        
