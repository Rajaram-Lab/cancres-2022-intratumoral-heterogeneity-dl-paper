"""
This file contains all functions used to extract nuclear features. The functions 
in this file will be called in Extract_Nuclear_Features.py

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

import os as os
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir
# from configparser import ConfigParser
# config = ConfigParser()
# config.read(os.path.join( os.getenv("HOME"),'.pythonConfig.ini'))
import sys
import yaml 
# import subprocess
# ROOT_DIR=subprocess.getoutput("git rev-parse --show-toplevel")
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
# sys.path.insert(0, os.path.join(ROOT_DIR,'HandCrafted/'))
sys.path.insert(0, os.path.join(ROOT_DIR,'External/StainTools/'))
#sys.path.insert(0, config.get('DL', 'dlCoreDir'))
# sys.path.insert(0, config.get('Code', 'stainTools'))
# import openslide as oSlide
#import ImageUtils as iu


import mahotas
import numpy as np
from scipy.stats import median_absolute_deviation,mode,kurtosis,skew
from scipy import ndimage as ndi
from skimage import morphology as morph
from skimage.morphology import skeletonize
# from skimage.transform import resize
from skimage import measure


import cv2 as cv2


#import progressbar
#import time
from abc import ABC,abstractmethod
from multiprocessing import Pool
from math import sqrt,pi as PI
from skimage.color import rgb2hed,rgb2lab
import colour

from staintools.stain_extraction.macenko_stain_extractor import MacenkoStainExtractor
from staintools.stain_extraction.vahadane_stain_extractor import VahadaneStainExtractor
from staintools.utils.optical_density_conversion import convert_OD_to_RGB
#from staintools.utils.optical_density_conversion import convert_RGB_to_OD
from staintools.utils.get_concentrations import get_concentrations

# %%

def ReadImg(blockCoords):
    (cornerPos,magLevel,blockSizeInt,isResizeNeeded,blockSizeOut,cornerPosOut)=blockCoords
    blockImg=np.array(mySlide.read_region(cornerPos,magLevel,blockSizeInt),order='f')[:,:,range(3)]
    if isResizeNeeded:
        blockOut=cv2.resize(blockImg,blockSizeOut)
    else:
        blockOut=blockImg
    return ((cornerPosOut,blockSizeOut),blockOut)


def ReadSlide(slide,blockWidth=2000,nWorkers=64,downSampleFactor=1,
              normalizer=None,downSampleTolerance=0.025):

    global mySlide
    mySlide=slide

    # Determine what level in the image pyramid we will pul images from
    # Not this is possibly an intermediate between highest magnification in pyrmaid and outpur mag desired
    if np.min(np.abs(np.array(slide.level_downsamples)-downSampleFactor))<\
        downSampleTolerance: #Is one of the existing levels close enough to desired downsampling
      
      # Use an existing downsampled imaged, no manual downsampling needed
      magLevel=int(np.argmin(np.abs(np.array(slide.level_downsamples)-downSampleFactor)))
      isResizeNeeded=False
      downSampleFactor=slide.level_downsamples[magLevel]

    else:
      # we will need to manually dowsnample
      magLevel=int(np.where(np.array(slide.level_downsamples)<downSampleFactor)[0][-1])
      isResizeNeeded=True
    
    downSampleExtracted=slide.level_downsamples[magLevel]

    dim=slide.dimensions
    # Output image size
    nR=int(np.ceil(dim[1]/downSampleFactor))
    nC=int(np.ceil(dim[0]/downSampleFactor))
     
    # Number of blocks needed
    nBR=int(np.ceil(nR/blockWidth))
    nBC=int(np.ceil(nC/blockWidth))

    # Detemine various image extraction parameters
    dSInt=downSampleFactor/downSampleExtracted # downsampling between pyramid level at which we extract image and output image
    
    # if self.isResizeNeeded:
    #   patchSizeLoaded=np.int32(np.round(np.array(patchSize)*self.downSampleFactor/self.downSampleExtracted))
    # else:
    #   patchSizeLoaded=blockWidth

    blockList=[]
    for r in range(nBR):
        for c in range(nBC):
            
            # CornerPosition for read_region in openSlide: expects in highest magLevel coords
            cornerPosOut=np.array([int(c*blockWidth),int(r*blockWidth)])
            cornerPos=np.uint32(cornerPosOut*downSampleFactor)
            
            # Image size for read_region: expects in coors at magLevel being read (i.e. intermediate)
            rSizeOut=int(min(blockWidth,(nR-cornerPosOut[1])))
            cSizeOut=int(min(blockWidth,(nC-cornerPosOut[0])))
            blockSizeOut=(cSizeOut,rSizeOut)
            rSizeInt=int(dSInt*rSizeOut)
            cSizeInt=int(dSInt*cSizeOut)
            blockSizeInt=(cSizeInt,rSizeInt)
            
            blockList.append((cornerPos,magLevel,blockSizeInt,isResizeNeeded,\
                              blockSizeOut,cornerPosOut))
            
    hImg=np.zeros((nR,nC,3),dtype=np.uint8)
    # bar=progressbar.ProgressBar(max_value=len(blockList))
    # for blockCounter,blockCoords in enumerate(blockList):
    #     blockImg=np.array(slide.read_region(blockCoords[0],0,blockCoords[1]))[:,:,range(3)]
    #     bar.update(blockCounter)
    # bar.finish()
    

    
    pool = Pool(processes=nWorkers,maxtasksperchild=100)    
    result=pool.map(ReadImg,blockList)
    for imgStuff in result:
        blockCoords,blockSize=imgStuff[0]
        img=imgStuff[1]
        boxSlice=np.s_[blockCoords[1]:((blockCoords[1]+blockSize[1])),
                       blockCoords[0]:((blockCoords[0]+blockSize[0])),
                       0:3]
        if normalizer is None:
            hImg[boxSlice]=img
        else:
            hImg[boxSlice]=normalizer(img)
    pool.close()
    pool.join()
    return hImg    

def SampleSlide(slide,numberOfPatches=250,patchSize=50,rgbThresh=220):                                                                                                                                                               
                                                                                                                                                                                                           
       lowResImg=np.array(slide.read_region((0,0),
                                            slide.level_count-1,
                                            slide.level_dimensions[slide.level_count-1]))[:,:,range(3)]
       
       lowResPos=np.where(np.any(lowResImg<rgbThresh,axis=-1))
       chosenIdx=np.random.choice(range(lowResPos[0].size),numberOfPatches,replace=True)             
       cX=np.int32(lowResPos[0][chosenIdx]*slide.level_downsamples[slide.level_count-1])                                                                                                                                              
       cY=np.int32(lowResPos[1][chosenIdx]*slide.level_downsamples[slide.level_count-1])
       pixelData=np.zeros((numberOfPatches*patchSize*patchSize,3))                                                         
       for n in range(numberOfPatches): #loop over selected patches and save to disk
           patchImg=np.asarray(slide.read_region((cY[n],cX[n]),0,(patchSize,patchSize)))[:,:,range(3)]
           pixelData[range(n*patchSize*patchSize,(n+1)*patchSize*patchSize),:]=np.resize(patchImg,(patchSize*patchSize,3))
       pixelData=np.uint8(pixelData[np.any(pixelData<rgbThresh,axis=-1),:])   
       isNotGreen=np.logical_and(pixelData[:,1]<1.0*pixelData[:,0],pixelData[:,1]<1.0*pixelData[:,2])
       pixelData=pixelData[isNotGreen,:]
       isNotDark=np.any(pixelData>50,axis=-1)
       pixelData=pixelData[isNotDark,:]
       return pixelData.reshape(pixelData.shape[0],1,3)    
     
    # %%



def GetH(x):
    return rgb2hed(x)[:,:,0]    
def GetLabL(x):
    return rgb2lab(x)[:,:,0]    
def GetHslL(x):
    return colour.RGB_to_HSL(x/255)[:,:,2]    
def Identity(x):
    return x

class Feature(ABC):
    @abstractmethod
    def __len__(self):
        pass
    @abstractmethod
    def names(self):
        pass
    @abstractmethod
    def profile(self,mask,imgList,objSlice):
        pass

class Location(Feature):
    def __len__(self):
        return 6
    def names(self):
        return ['Y_Start','Y_End','X_Start','X_End','Centroid_Y','Centroid_X']
    def profile(self,mask,img,objSlice):
        yStart=objSlice[0].start
        yEnd=objSlice[0].stop
        xStart=objSlice[1].start
        xEnd=objSlice[1].stop
        centroid=np.mean(np.nonzero(mask),axis=1)
        return np.array([yStart,yEnd,xStart,xEnd,centroid[0]+yStart,centroid[1]+xStart])
    

class Size(Feature):
    def __len__(self):
        return 3
    def names(self):
        return ['Area','BBox_Area','Equivalent_Diameter']
    def profile(self,mask,img,objSlice):
        area= np.sum(mask)
        eqDiameter=sqrt(4 * area / PI)
        return np.array([area,mask.size,eqDiameter])
    
class Shape(Feature):
    def __len__(self):
        return 2
    def names(self):
        return ['Eccentricity','Major_Axis_Length']
    def profile(self,mask,imgList,objSlice):
        l1, l2 = measure._moments.inertia_tensor_eigvals(mask)
        if l1 == 0:
            eccentricity=0
        else :
            eccentricity=sqrt(1-l2/l1)
        majAxLength=4*sqrt(l1)
        return np.array([eccentricity,majAxLength])

class Convexity(Feature):
    def __len__(self):
        return 2
    def names(self):
        return ['Convex_Area','Solidity']
    def profile(self,mask,imgList,objSlice):
        convexHull=morph.convex_hull_image(mask)
        convexArea=np.sum(convexHull)
        solidity=mask.size/convexArea
        return np.array([convexArea,solidity])    
    
class Skeleton(Feature):
    def __len__(self):
        return 1
    def names(self):
        return ['Skeleton_Length']
    def profile(self,mask,imgList,objSlice):
        skeleton=skeletonize(mask)
        return np.sum(skeleton)    

class IntensityStats(Feature):
    def __init__(self,imgNum,featPrefix,transform=Identity):
        self.imgNum=imgNum
        self.featPrefix=featPrefix
        self.transform=transform  
        
    def __len__(self):
        return 8
    def names(self):
        suffixes=['_Mean','_Median','_Std','_MAD','_Min','_Max','_Kurtosis','_Skewness']
        return [self.featPrefix+s for s in suffixes]
    def profile(self,mask,imgList,objSlice):
        imgVals=np.float32(self.transform(imgList[self.imgNum])[mask])
        
        return np.array([np.mean(imgVals),np.median(imgVals),
                         np.std(imgVals),median_absolute_deviation(imgVals),
                         np.min(imgVals),np.max(imgVals),kurtosis(imgVals),
                         skew(imgVals)])        

class Haralick_Texture(Feature):
    def __init__(self,imgNum,minVal,maxVal,featPrefix,transform=Identity):
        self.imgNum=imgNum
        self.featPrefix=featPrefix
        self.transform=transform  
        self.minVal=minVal
        self.maxVal=maxVal
        
    def __len__(self):
        return 13
    def names(self):
        n=[]
        featNames1=['2nd_moment','contrast','correlation','variance',
                            'inv_diff_moment','sum_avg','sum_variance',
                            'sum_entropy','entropy','diff_var','diff_entropy',
                            'inf_corr1','inf_corr2']    

        
        for featNum in np.arange(13):
            n.append(self.featPrefix+'_Haralick_'+featNames1[featNum])
                
                
                
        return n
    def profile(self,mask,imgList,objSlice):
        imgVals=(self.transform(imgList[self.imgNum])-self.minVal)/(self.maxVal-self.minVal)
        
        imgVals[imgVals<0]=0
        imgVals[imgVals>1]=1
        imgValsInt=np.uint8(255*imgVals)
        imgValsInt[~mask]=0
        try:
            haralickAll=np.concatenate(mahotas.features.haralick(imgValsInt,ignore_zeros=True))
            haralick=np.mean(haralickAll.reshape((4,13)),axis=0)
        except:
            haralick=np.zeros(13)
        return haralick     


class Label(Feature):
    def __init__(self,imgNum,featPrefix):
        self.imgNum=imgNum
        self.featPrefix=featPrefix
        
    def __len__(self):
        return 1
    def names(self):
        suffixes=['_label']
        return [self.featPrefix+s for s in suffixes]
    def profile(self,mask,imgList,objSlice):
        
        
        return mode(imgList[self.imgNum][mask].flatten())[0][0]

class FeatureExtractor():
    def __init__(self,featureList):
        nFeat=0
        if not all([isinstance(f,Feature) for f in featureList]):
            raise SystemError('Passed classes must be of class Feature')
        self.featureNames=['label']
        for f in featureList:
            if len(f) != len(f.names()) :
               raise SystemError('Wrong Size in '+ f.__name__())
            nFeat=nFeat+len(f)
            self.featureNames=self.featureNames+f.names()
        self.numberOfFeatures=nFeat            
        self.fList=featureList
    
    def Run(self,inputs)   :
        label,mask,img,objSlice=inputs
        outMat=np.zeros(self.numberOfFeatures+1)
        outMat[0]=label
        counter=1
        for i,f in enumerate(self.fList):
            outMat[counter:counter+len(f)]=f.profile(mask,img,objSlice)
            counter+=len(f)
        return outMat    
    
    def Names(self):
        return [f.__name__ for f in self.fList]
     


def ExtractFeatures(mask,featureList,imgList=[]): 
    #mask is the matrix which defines the objects
    # featureList is a list of objects of type Feature 
    # imgList is a list of images corresponding to mask (e.g. H&E image)
    nObj, labelMat, stats, centroids=cv2.connectedComponentsWithStats(np.uint8(mask),8,cv2.CV_32S)
    objects=ndi.measurements.find_objects(labelMat)
    dataList=[]
    for objNum,objSlice in enumerate(objects):
        temp=labelMat[objSlice]
        label=objNum+1
        objMask=temp.copy()==label
        iList=[]
        for i in range(len(imgList)):
            iList.append(imgList[i][objSlice])
        dataList.append((label,objMask,iList,objSlice)) # this contains all the inputs for a single object
        #proc=Process(target=Area,args=(objMask,))
        #procs.append(proc)
        #proc.start()
    pool = Pool(processes=64,maxtasksperchild=100)    
    
    featCalc=FeatureExtractor(featureList)  
    result=pool.map(featCalc.Run,dataList)
    pool.close()
    pool.join()
        
    featureMat=np.zeros((len(result),result[0].size))
    for f in result:
        featureMat[int(f[0]-1),:]=f
    featureNames=featCalc.featureNames        
    return featureMat,featureNames
