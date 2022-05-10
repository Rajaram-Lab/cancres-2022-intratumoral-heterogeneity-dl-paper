"""
This file contains all functions used to perform normalization on TMA samples.

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

sys.path.insert(0,os.path.join(ROOT_DIR, 'External/StainTools/'))

import numpy as np
from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.utils.get_concentrations import get_concentrations


# %%  Various functions written for normalization      
def SampleSlide(slide,mask,numberOfPatches=250,patchSize=50,flatten=True):                                                                                                                                                               
                                                                                                                                                                                                           
       downSampleFactor=np.sqrt(np.product(slide.dimensions)/mask.size)
       lowResPos=np.where(mask)
       if(len(lowResPos[0])>numberOfPatches):
           chosenIdx=np.random.choice(range(lowResPos[0].size),numberOfPatches,replace=False)       
       else:
           chosenIdx=np.random.choice(range(lowResPos[0].size),numberOfPatches,replace=True)
       cX=np.int32(lowResPos[0][chosenIdx]*downSampleFactor)                                                                                                                                              
       cY=np.int32(lowResPos[1][chosenIdx]*downSampleFactor)
       if flatten:
           pixelData=np.zeros((numberOfPatches*patchSize*patchSize,3))    
       else:                                                     
           pixelData=np.zeros((numberOfPatches,patchSize,patchSize,3),order='f')
       for n in range(numberOfPatches): #loop over selected patches and save to disk
           patchImg=np.asarray(slide.read_region((cY[n],cX[n]),0,(patchSize,patchSize)))[:,:,range(3)]
           if flatten:
               pixelData[range(n*patchSize*patchSize,(n+1)*patchSize*patchSize),:]=np.resize(patchImg,(patchSize*patchSize,3))
           else:
               pixelData[n]=patchImg
       if flatten:        
           return np.uint8(pixelData.reshape(pixelData.shape[0],1,3))
       else:
           return np.uint8(pixelData)

    

class StainNormalizer(object):

    def __init__(self, method):
        if method.lower() == 'macenko':
            self.extractor = MacenkoStainExtractor
        elif method.lower() == 'vahadane':
            self.extractor = VahadaneStainExtractor
        else:
            raise Exception('Method not recognized.')

    def fit(self, targetSlide,targetMask,sourceSlide,sourceMask,percentileCutoff=99):
        """
        Fit to a target image.
        :param target: Image RGB uint8.
        :return:
        """
        if isinstance(targetSlide,np.ndarray) and targetMask is None:
            target=targetSlide
        else:
            target=SampleSlide(targetSlide,targetMask)
        if isinstance(sourceSlide,np.ndarray) and sourceMask is None:
            source=sourceSlide
        else:
            source=SampleSlide(sourceSlide,sourceMask)
        self.stain_matrix_target = self.extractor.get_stain_matrix(target)
        self.target_concentrations = get_concentrations(target, self.stain_matrix_target)
        self.maxC_target = np.percentile(self.target_concentrations, percentileCutoff, axis=0).reshape((1, 2))
        
        self.stain_matrix_source = self.extractor.get_stain_matrix(source)
        self.source_concentrations = get_concentrations(source, self.stain_matrix_source)
        self.maxC_source = np.percentile(self.source_concentrations, percentileCutoff, axis=0).reshape((1, 2))
        

    def transform(self, I):
        """
        Transform an image.
        :param I: Image RGB uint8.
        :return:
        """
        source_concentrations = get_concentrations(I, self.stain_matrix_source)
        source_concentrations *= (self.maxC_target / self.maxC_source)
        tmp = 255 * np.exp(-1 * np.dot(source_concentrations, self.stain_matrix_target))
        return tmp.reshape(I.shape).astype(np.uint8)
    
    def transformFull(self, I):
        """
        Transform a collection of images.
        :param I: Image RGB float32.
        :return:
        """
        if len(I.shape)>3:
            out=np.zeros(I.shape,dtype=np.float32)
            for i in np.arange(out.shape[0]):
                out[i]=self.transform(I[i])/255.0
        else:
            out=self.transform(I)/255.0
        return out    

def GetCounts(classImg,numberOfClasses):
    counts=np.zeros(numberOfClasses)
    uniqueVals, counts = np.unique(classImg, return_counts=True)
    for i in range(len(uniqueVals)):
            if uniqueVals[i]<numberOfClasses:
                counts[uniqueVals[i]]=counts[i]
    return counts        

def DetermineMasks(targetClasses,sourceClasses,targetClassNames,
                   sourceClassNames=None,classOrder=['stroma','normal']):
    if(sourceClassNames is None):
        sourceClassNames=targetClassNames
    minCounts=1E5
    sourceCounts=GetCounts(sourceClasses,len(sourceClassNames))
    targetCounts=GetCounts(targetClasses,len(targetClassNames))
    for chosenClass in classOrder:
        if chosenClass in sourceClassNames and chosenClass in targetClassNames:
            sourceClassNumber=sourceClassNames.index(chosenClass)
            targetClassNumber=targetClassNames.index(chosenClass)
            if sourceCounts[sourceClassNumber]>minCounts and targetCounts[targetClassNumber]>minCounts:
                print('Using ' +chosenClass)
                return targetClasses==targetClassNumber,sourceClasses==sourceClassNumber
    print('Using Tumor')
    return targetClasses==targetClassNames.index('tumor'),sourceClasses==sourceClassNames.index('tumor')
