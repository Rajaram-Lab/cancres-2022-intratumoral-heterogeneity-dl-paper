#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LocalizedLoss_PatchGeneration is used to generate the patches for the localized
loss samples of WSI. These samples require manual annotations to distinguish the 
loss regions and the wildtype regions in the tissue.

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
with open(os.path.join(ROOT_DIR,'Parameters/Training_Params.yaml')) as file:
    trainParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
dataDir = projectPaths['DataDir']


import Util.PatchGen as pg
import openslide as oSlide

from skimage.transform import resize
import numpy as np
import pickle
from glob import glob


"""
Note: For coding purposes, localized loss cases are referred to simply to as 
    'Focal'. Any variables with the 'Focal' reference will be refering the the 
    localized loss status.
"""

  #%% Functions
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical
  
def ResizeOneHotEncode(patchMask,classes,dimensions):
    """
    Function that resizes masks using a one hot encode method 
    Inputs: 
        patchMask : array : patch from mask 
        classes : list : zipped class list. I.e [0:'background',1:'blood'..etc] 
        patchSize : int : default is 224
    Outputs: 
        finalPatchMask : array : resized patch mask to be same size as original patches
    """
    patchMaskOHE=to_categorical(patchMask,num_classes=len(classes))
    resizedPatchMask=np.zeros((dimensions[0],dimensions[1],len(classes)))
    for layer in range(len(classes)):
        resizedPatchMask[:,:,layer]=resize(patchMaskOHE[:,:,layer],(dimensions[0],dimensions[1]))
    finalPatchMask=np.argmax(resizedPatchMask,axis=2)
    return finalPatchMask         


  
#%%
from scipy import ndimage as ndi


def PatchesFromMaskFocal(slide,mask,downSampleFactors,patchSizes,maskToClassDict,
                    maxPatchesPerAnno=1000,maxAvgPatchOverlap=0.9,minFracPatchInAnno=0,
                    showProgress=False):
  """ Generate patch data given a mask
  Inputs:
  slide -         an openSlide slide
  mask -          a numpy array with element values denoting the classes of different pixels 
                  in the slide. Size must be proportional to that of slide                   
  downSampleFactors - a 1-D list/numpy array denoting the scale relative to max-magnification
                      e.g., [1,4] would denote we wanted patches at 20X and 5X (assuming max-mag was 20X)                    
  patchSizes    - a 1D list/np array denoting patch size in pixels at the scales indicated
                  by downSampleFactors
  maskToClassDict - A dictionary indicating the class (a string) that each value in the mask
                    corresponds to. Note: only patches corresponding to values 
                    present in this dict will be generated.               
  """
  patchReaders={}
  patchCenters=[]
  patchClasses=[]
  for dS in np.unique(downSampleFactors).tolist():
    patchReaders[dS]=pg.PatchReader(slide,dS)
    
  # How many fold smaller is the mask than the image
  maskToImgDownScale=((slide.dimensions[0]/mask.shape[1])+(slide.dimensions[1]/mask.shape[0]))/2
  # Patch sizes in pixels at the max-mag level (e.g. 64px patch at 5X=256px at 20X)
  absPatchSizes=np.array(downSampleFactors)*np.array(patchSizes)
  # Largest absolute patch size scaled down to the mask (used to determine boundaries)
  maxPatchSizeScaled=np.int32(np.ceil(np.max(absPatchSizes)/maskToImgDownScale))
  
  totalNumberOfPatches=0
   # for annoNum in maskToClassDict: # Loop over annotations
  #startTime=time.time()
    
  if(minFracPatchInAnno>0): # Do we need to determine %of patch in class
    # Run uniform filter with kernel-size equal to biggest patch size to determine patch composition  
    # candidateMaskClass=ndi.uniform_filter(np.float32(mask==2),maxPatchSizeScaled)>minFracPatchInAnno
    candidateMaskClass=mask==2
    candidateMask=ndi.uniform_filter(np.float32(mask!=0),maxPatchSizeScaled)>minFracPatchInAnno
    candidateMask=np.logical_and(candidateMask,mask!=0) 
    
  # Exclude borders of image
  candidateMask[range(int(maxPatchSizeScaled/2)),:]=False
  candidateMask[range(0,-int(maxPatchSizeScaled/2),-1),:]=False
  candidateMask[:,range(int(maxPatchSizeScaled/2))]=False
  candidateMask[:,range(0,-int(maxPatchSizeScaled/2),-1)]=False
  
  
  # These are potential positions for the patch center
  candidatePos=np.where(candidateMask)

  
  # Determine number of patches to extract
  maxPatchesForOverlap=np.int32(np.sum(candidateMask)*(maxAvgPatchOverlap+1)/(maxPatchSizeScaled*maxPatchSizeScaled))
  numberOfPatches=min(maxPatchesPerAnno,maxPatchesForOverlap)
    
  if numberOfPatches>0:
    chosenIdx=np.random.choice(candidatePos[0].size,numberOfPatches)
    pC=np.zeros((numberOfPatches,2))  
    # Rescale from mask to slide coords
    pC[:,1]=candidatePos[0][chosenIdx]*maskToImgDownScale
    pC[:,0]=candidatePos[1][chosenIdx]*maskToImgDownScale
    # Add random stagger
    if(maskToImgDownScale>1):
      pC=pC+ np.random.randint(low=0,high=np.round(maskToImgDownScale/2),size=(numberOfPatches,2))- np.round(maskToImgDownScale/2)
    patchCenters.append(np.int32(pC))
    
    # patchClasses=patchClasses+[maskToClassDict[annoNum]]*numberOfPatches
    patchClasses=candidateMaskClass[candidatePos[0][chosenIdx],candidatePos[1][chosenIdx]]
    totalNumberOfPatches=totalNumberOfPatches+numberOfPatches

  if(totalNumberOfPatches>0):
    patchCenters=np.concatenate(patchCenters)
    
    patchData=[]
    for scale in np.arange(np.array(downSampleFactors).size): # Loop over scales
      dS=downSampleFactors[scale]
      pS=patchSizes[scale]
      patchData.append(patchReaders[dS].LoadPatches(patchCenters,np.array([pS,pS]),showProgress))
    return patchData,patchClasses,patchCenters
  else:      
    return [],[],[]

# %% Select files to work with

xmlDir=os.path.join(ROOT_DIR, projectPaths['Region_Level']['XmlDir'])
svsDir=os.path.join(dataDir, projectPaths['Data']['ImageData'],'WSI/')
saveDir=os.path.join(dataDir,projectPaths['Data']['PatchData'],'WSI-LL/')
tumorDir=os.path.join(dataDir,projectPaths['Tumor'],'WSI/')

xmlFiles=glob(os.path.join(xmlDir,'*.xml'))
tumorMaskFiles=[f.replace(xmlDir,tumorDir).replace('xml','pkl') for f in xmlFiles]
svsFiles=[f.replace(xmlDir,svsDir).replace('xml','svs') for f in xmlFiles]
hdf5Files=[f.replace(xmlDir,saveDir).replace('xml','hdf5') for f in xmlFiles]


#%% Patch Generation for focal cases 
for fileCounter in range(len(xmlFiles)):   
    slide=oSlide.open_slide(svsFiles[fileCounter])
    # slideImage=np.array(slide.read_region((0,0),1,slide.level_dimensions[1]))[:,:,range(3)]
    with open(tumorMaskFiles[fileCounter],'rb') as handle:
      maskData=pickle.load(handle)

    tumorMask=np.uint8(maskData[0]==maskData[1].index('tumor'))
    
    downSampleFactor=slide.level_dimensions[0][0]/tumorMask.shape[1]
    
    annoMask, maskDict= pg.MaskFromXML(xmlFiles[fileCounter],'Focal_BAP1',slideDim=(slide.level_dimensions[0][0],slide.level_dimensions[0][1]),
                                       downSampleFactor=downSampleFactor,distinguishAnnosInClass=False)
    
    lossMask=ResizeOneHotEncode(annoMask,classes=['background','BAP1Neg'], dimensions=tumorMask.shape)

    mask=np.uint8(tumorMask)*2
    mask[np.logical_and(lossMask!=0,mask)]=1
    
    downSampleLevels=[1,4] # Downsampling factor relative to max (typically 20X). So 4 will give the 5X image. Adding multiple values gives patches at different scales
    patchSizeList=[224,224] # Patch size (we assume patches are square) in pixels. Specify pcdatch size separately for each scale in downSampleLevels
    showProgress=False
    maxPatchesPerAnno=1200 # Maximum number of patches sampled from an annotation
    maxAvgPatchOverlap=8.0 # How tightly patches are allowed to overlap. 0 implies no overlap, 1 implies number of patches is selected so that combined area of patches= area of annotation
    minFracPatchInAnno=0.8 
    
    maskToClassDict={1:'neg',2:'pos'}
    
    patchData,patchClasses,patchCenters=PatchesFromMaskFocal(slide,mask,
                                                          downSampleLevels,patchSizeList,
                                                          maskToClassDict,
                                                          maxPatchesPerAnno=maxPatchesPerAnno,
                                                          showProgress=showProgress,
                                                          maxAvgPatchOverlap=maxAvgPatchOverlap,
                                                          minFracPatchInAnno=minFracPatchInAnno)
    if not os.path.exists(hdf5Files[fileCounter]):
        pg.SaveHdf5Data(hdf5Files[fileCounter],patchData,patchClasses,patchCenters,downSampleLevels,patchSizeList,svsFiles[fileCounter])   
        print(hdf5Files[fileCounter],' done!')
    else: 
        print(hdf5Files[fileCounter],' Already Exists!')