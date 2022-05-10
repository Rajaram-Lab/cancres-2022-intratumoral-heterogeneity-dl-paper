"""
PatchGen contains all of the functions used to generate patches from slide images.

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

import h5py, re, progressbar
import numpy as np

from PIL import Image, ImageDraw
from scipy import ndimage as ndi

import Util.ImageUtils as iu

# %% Functions To Read Patches
class PatchReader():
    # Convenience Class to Read Image Regions With Arbitrary Magnification from a Slide
    def __init__(self,slide,downSampleFactor,downSampleTolerance=0.025):
        # Inputs:
        # slide - openslide slide that image regions need to be sampled from
        # downSampleFactor - number specifying the magnfication level relative 
        #                    to the highest resolution available. e.g. downSampleFactor
        #                    of 3 would yield patches with one third the max resolution
        # downSampleTolerance - a number specifying numerical accuracy of the downSampleFactor.
        #                    It is used to select a pre-existing maglevel. For example, 
        #                    if it is 0.025 and a downsampleFactor or 4 was used, 
        #                    and a layer with 4.024 was available, it would be used
        self.slide=slide
        assert downSampleFactor>=1 # cannot get higher resolution than original image
        self.downSampleFactor=downSampleFactor
        
        if np.min(np.abs(np.array(slide.level_downsamples)-downSampleFactor))<downSampleTolerance:
          self.magLevel=int(np.argmin(np.abs(np.array(slide.level_downsamples)-downSampleFactor)))
          self.isResizeNeeded=False
          self.downSampleExtracted=self.downSampleFactor

        else:
          self.magLevel=int(np.where(np.array(slide.level_downsamples)<downSampleFactor)[0][-1])
          self.isResizeNeeded=True
          self.downSampleExtracted=slide.level_downsamples[self.magLevel]
  
    
    def LoadPatches(self,patchCenters,patchSize,showProgress=False):
      # Generate patch data given patch centers and patch sizes
      # Inputs:
      # patchCenters - a numpy array, with 2 columns (specifying xy positions 
      #                in ABSOLUTE coordinates, i.e. at the max mag level). 
      #                Rows correspoind to different patches
      # patchSize - a 1x2 numpy array/list/tuple specifying OUTPUT patch size 
      #             (i.e. at the output downSample space)
      
      assert ((type(patchCenters) is np.ndarray) and patchCenters.shape[1]==2),sys.error('Invalid Patch Centers')
      numberOfPatches  =patchCenters.shape[0]
      
      if self.isResizeNeeded:
        patchSizeLoaded=np.int32(np.round(np.array(patchSize)*self.downSampleFactor/self.downSampleExtracted))
      else:
        patchSizeLoaded=patchSize
      
      patchData=np.zeros((numberOfPatches,patchSize[0],patchSize[1],3),np.uint8)
      
      cornerPositions=np.int32(np.floor(np.array(patchCenters)-self.downSampleExtracted*(patchSizeLoaded+1)/2.0))
      if showProgress:
        bar=progressbar.ProgressBar(max_value=numberOfPatches)
      for patchCounter in range(numberOfPatches):
        img=self.slide.read_region(cornerPositions[patchCounter],self.magLevel,patchSizeLoaded)
        if self.isResizeNeeded:
          img=img.resize(patchSize,Image.BILINEAR)
        patchData[patchCounter]=np.array(img,np.uint8)[:,:,np.arange(3)]
        if showProgress:
          bar.update(patchCounter)
      if showProgress:  
        bar.finish()      
      
      return patchData

# %% Core function to generate patches given a mask
def PatchesFromMask(slide,mask,downSampleFactors,patchSizes,maskToClassDict,
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
    patchReaders[dS]=PatchReader(slide,dS)
    
  # How many fold smaller is the mask than the image
  maskToImgDownScale=((slide.dimensions[0]/mask.shape[1])+(slide.dimensions[1]/mask.shape[0]))/2
  # Patch sizes in pixels at the max-mag level (e.g. 64px patch at 5X=256px at 20X)
  absPatchSizes=np.array(downSampleFactors)*np.array(patchSizes)
  # Largest absolute patch size scaled down to the mask (used to determine boundaries)
  maxPatchSizeScaled=np.int32(np.ceil(np.max(absPatchSizes)/maskToImgDownScale))
  
  totalNumberOfPatches=0
  for annoNum in maskToClassDict: # Loop over annotations
    #startTime=time.time()
    
    if(minFracPatchInAnno>0): # Do we need to determine %of patch in class
      # Run uniform filter with kernel-size equal to biggest patch size to determine patch composition  
      candidateMask=ndi.uniform_filter(np.float32(mask==annoNum),maxPatchSizeScaled)>minFracPatchInAnno
    else:
      candidateMask=mask==annoNum

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

      patchClasses=patchClasses+[maskToClassDict[annoNum]]*numberOfPatches
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


# %% Function to generate Mask from XML
def MaskFromXML(annoFile,layerName,slideDim,downSampleFactor=1,distinguishAnnosInClass=True,
                outerBoxLabel='square',outerBoxClassName='BG'):  
  """ Generate a mask from XML (aperio) or TXT (QuPath) annotation
  Inputs:
  annoFile -      string denoting filename of txt/xml file 
  layerName -     for XML files, this is the annotation layerName that is used.
                  Unused for txt                          
  slideDim -      tuple denoting height/width of the slide in pixels at max magnification                    
  downSampleFactor    - scalar dnoting how may fold smaller the mask is than the slide.
                      This can be useful to increase speed.
  distinguishAnnosInClass - Boolean Scalar. If True all annotations from the same class
                  will be merged into a single annotation    
           
  """  
  outputDim=tuple(np.int32(np.array(slideDim)/downSampleFactor))
  if(annoFile.endswith('.xml')):
    annoNames,subAnno=iu.GetXmlAnnoNames(annoFile)
    layersToUse=[i for i,x in enumerate(annoNames) if x==layerName]
  else:
    layersToUse=[0]
  for annoLayer in layersToUse:
   
    if(annoFile.endswith('.xml')):
      annos=iu.ReadRegionsFromXML(annoFile,layerNum=annoLayer)
      regionPos=annos[0]
      regionNames=annos[1]
      regionInfo=annos[2]
    else:
      regionPos,regionNames,regionInfo,isNegative=iu.GetQPathTextAnno(annoFile)
   
    outerBoxIdx=np.where([(bool(re.compile(r"^[+-]\s*").match(i)) and (i == '+'+outerBoxLabel)) for 
                                       i in regionNames])[0]
    validNameIdx=np.where([(bool(re.compile(r"^[+-]\s*").match(i)) and not(i == '+'+outerBoxLabel)) for 
                                       i in regionNames])[0]
    validNames=np.array(regionNames)[validNameIdx]                    

    validIsPos=[s[0]=="+" for s in validNames]
    validSuffixes=[s[1:] for s in validNames]
    #print(validSuffixes)      
    
    if(distinguishAnnosInClass):
      numberOfAnnos=np.sum(validIsPos)
    else:
      numberOfAnnos=len(np.unique(validSuffixes))
      nameToNum={}
      for num,name in enumerate(np.unique(validSuffixes)):
        nameToNum[name]=num+1
        
    maskToClassDict={}
    if(numberOfAnnos<254):
       mask = Image.new("P",outputDim, 0)
    else:
       mask = Image.new("I",outputDim, 0)
    
    regionOrder=np.concatenate([validNameIdx[np.where(validIsPos)[0]],
                                 validNameIdx[np.where(np.logical_not(validIsPos))[0]]] )
    validSuffixes=np.array(validSuffixes)
    validIsPos=np.array(validIsPos)
    namesOrdered=np.concatenate([validSuffixes[np.where(validIsPos)[0]],
                                 validSuffixes[np.where(np.logical_not(validIsPos))[0]]] )
    isPosOrdered=np.concatenate([validIsPos[np.where(validIsPos)[0]],
                                 validIsPos[np.where(np.logical_not(validIsPos))[0]]] )
    
    for regionCounter,regionIdx in enumerate(regionOrder):
      
      pos=regionPos[regionIdx]/downSampleFactor
      name=namesOrdered[regionCounter]
      isPos=isPosOrdered[regionCounter]
      
      poly=np.array(pos)
      poly=np.concatenate((poly,np.expand_dims(poly[0,:],axis=0)))   
      if isPos:
        if(distinguishAnnosInClass):
            annoClass=regionCounter+1
        else:
            annoClass=nameToNum[name]
            #print(isPos,annoClass)
        maskToClassDict[annoClass]=name
        ImageDraw.Draw(mask).polygon(poly.ravel().tolist(), outline=int(annoClass), fill=int(annoClass))
      else: 
        ImageDraw.Draw(mask).polygon(poly.ravel().tolist(), outline=0, fill=0) # Should change to only erase annotations with same label (Currently it sets those pixels to 0)
    mask=np.array(mask)
    
    if(len(outerBoxIdx)>0):
      if(numberOfAnnos<254):
        outerMask = Image.new("P",outputDim, 0)
      else:
        outerMask = Image.new("I",outputDim, 0)
      for oIdx in outerBoxIdx:
        pos=regionPos[oIdx]/downSampleFactor
        poly=np.array(pos)
        poly=np.concatenate((poly,np.expand_dims(poly[0,:],axis=0)))   
        ImageDraw.Draw(outerMask).polygon(poly.ravel().tolist(), outline=int(numberOfAnnos+1), fill=int(numberOfAnnos+1))
      outerMask=np.array(outerMask)
      mask[np.logical_not(outerMask>0)]=0
      mask[np.logical_and(outerMask>0,mask==0)]=numberOfAnnos+1
      maskToClassDict[numberOfAnnos+1]= outerBoxClassName
    
    return mask,maskToClassDict                  

  

# %% HDF5 Saving and Loading


def SaveHdf5Data(hdf5Filename,patchData,patchClasses,patchCenters,downSampleList,patchSizeList,analyzedFile):
  with h5py.File(hdf5Filename, 'w') as f:
      patches=f.create_group('patches')
      patches.attrs['numberOfScales']=len(patchData)
      patches.attrs['numberOfPatches']=len(patchClasses)
      patches.attrs['analyzedFile']=analyzedFile
      for scaleNumber,p in enumerate(patchData):
        d=patches.create_dataset(str(scaleNumber), data=patchData[scaleNumber])
        d.attrs['patchSize']=patchSizeList[scaleNumber]
        d.attrs['downSampleFactor']=downSampleList[scaleNumber]
        
      f.create_dataset('classes', data=np.array(patchClasses,dtype='S'))
      f.create_dataset('patchCenters', data=patchCenters)


    
def LoadHdf5Data(hdf5Filename):    
  hdf5Data={}    
  with h5py.File(hdf5Filename, 'r') as f:
    hdf5Data['classLabels']=np.array(f['classes'][:],'U')
    hdf5Data['patchCenters']=f['patchCenters'][:]
    for k in f['patches'].attrs.keys():
      hdf5Data[k]=f['patches'].attrs[k]
    hdf5Data['patchData']=[]
    for s in np.arange(hdf5Data['numberOfScales']):
      temp={}
      temp['patches']=f['patches'+'/'+str(s)][:]
      temp['patchSize']=f['patches'+'/'+str(s)].attrs['patchSize']
      temp['downSampleFactor']=f['patches'+'/'+str(s)].attrs['downSampleFactor']
      hdf5Data['patchData'].append(temp)
  return hdf5Data    

# %% Load Data by combing from a list of hdf5 files

def LoadPatchData(hdf5FileList,classDict=None,returnSampleNumbers=False,returnPatchCenters=False):
    if len(hdf5FileList) == 0:
      print("Empty hdf5 list supplied. Exiting... \n")
      os._exit(1)
    # Find number of scales/patch dimensions
    numberOfPatches=0

    if classDict is None:
      uniqueLabels=set()
    else:
      uniqueLabels=np.array([k for k in classDict.keys()])
    
    patchSizes=[]
    for fileCounter,hdf5File in enumerate(hdf5FileList):
      with h5py.File(hdf5File, 'r') as f:
        if fileCounter==0:
          numberOfScales=f['patches'].attrs['numberOfScales']
          for s in np.arange(numberOfScales):
            patchSizes.append(f['patches'+'/'+str(s)].attrs['patchSize'])
              
        classes=np.array(f['classes'][:],'U')
        if classDict is None:
          numberOfPatches=numberOfPatches+f['patches'].attrs['numberOfPatches']
          uniqueLabels.update(set(classes))
        else:
          isInDict= np.array([s in uniqueLabels for s in classes])
          numberOfPatches=numberOfPatches+np.sum(isInDict)
       
    if classDict is None:
      classDict={}
      for num,name in enumerate(np.sort(list(uniqueLabels))):
        classDict[name]=num

    # Change variable type to int
    numberOfPatches = int(numberOfPatches)
    
    #print(np.array(uniqueLabels))     
    patchData=[]
    #preAllocate patches across scales and classes 
    patchClasses=np.zeros((numberOfPatches))
    sampleNumbers=np.zeros((numberOfPatches))
    patchCenters=np.zeros((numberOfPatches,2))

    for scale in range(numberOfScales):
      print(numberOfPatches,patchSizes[scale],numberOfScales)
      patchData.append(np.zeros((numberOfPatches,patchSizes[scale],patchSizes[scale],3),dtype=np.uint8))
    #print('Initialized data in '+ str(time.time() - startTime)+' seconds')
    patchCounter=0  
    bar=progressbar.ProgressBar(max_value=len(hdf5FileList))
    for fileCounter,hdf5File in enumerate(hdf5FileList):
      
      hdf5Data=LoadHdf5Data(hdf5File)
      isInDict= np.array([s in uniqueLabels for s in hdf5Data['classLabels']])
      nValidPatches=np.sum(isInDict)
      if(nValidPatches>0):
      #print(nValidPatches)
          try:
              patchClasses[np.uint32(np.arange(patchCounter,patchCounter+nValidPatches))]=np.array([classDict[c] for c in hdf5Data['classLabels'][isInDict]])
          except Exception as e:
              print(nValidPatches)
              print(np.arange(patchCounter,patchCounter+nValidPatches))
              print(e)
              sys.exit()
          sampleNumbers[np.uint32(np.arange(patchCounter,patchCounter+nValidPatches))]=fileCounter
          patchCenters[np.uint32(np.arange(patchCounter,patchCounter+nValidPatches)),:]=hdf5Data['patchCenters'][isInDict]
          for scale in range(numberOfScales):
            patchData[scale][np.uint32(np.arange(patchCounter,patchCounter+nValidPatches)),:,:,:]=hdf5Data['patchData'][scale]['patches'][isInDict]
            #print(hdf5Data['patchData'][scale]['patches'][isInDict])
      patchCounter=patchCounter+nValidPatches  
      bar.update(fileCounter)
    bar.finish()   
    if returnPatchCenters:
      return patchData, patchClasses,classDict,sampleNumbers,patchCenters
    elif returnSampleNumbers:
      return patchData, patchClasses,classDict,sampleNumbers
    else:
      return patchData, patchClasses,classDict 
      
