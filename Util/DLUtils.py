"""
DLUtils contains all of the core DL functions used in this project.

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

import os,  sys
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir
sys.path.insert(0, ROOT_DIR)

import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence

import pickle
import progressbar
import time
# %% Importing Functions
def isIntersecting1d(a_min,a_max,b_min,b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)

def isIntersecting(bb1,bb2):
    return isIntersecting1d(bb1[0],bb1[2],bb2[0],bb2[2])  and isIntersecting1d(bb1[1],bb1[3],bb2[1],bb2[3])
               

def Get_Eff_Kernel_Params(model):
    nLayers=len(model.layers)
    k=-1
    s=-1
    for layerNum in range(nLayers):
        layerConfig=model.layers[layerNum].get_config()
        if('kernel_size' in layerConfig):
            k1=layerConfig['kernel_size'][0]
        else:
            k1=1
        if('strides' in layerConfig):
            s1=layerConfig['strides'][0]
        else:
            s1=1
            
        #print(layerNum,k1,s1)
        if(k>0 and s>0):    
            k=k+(k1-1)*s
            s=s1*s
        else:
            k=k1
            s=s1
        
    return k,s


class Classifier():
    
    def __init__(self):
        self.model=[]
        self.patchSize=[]
        self.magLevel=[]
        self.labelNames=[]
        self.intModel=[]
        self.numberOfClasses=[]
        self.effectiveKernel=[]
        self.effectiveStride=[]
        

    def Init(self,model, patchSize, magLevel,labelNames,savefile='classifier.pkl'):

        self.model=model
        self.patchSize=patchSize
        self.magLevel=magLevel
        self.labelNames=labelNames
    
        self.intModel=Model(inputs=model.input,outputs=model.get_layer(index=-2).output)
        self.numberOfClasses=self.intModel.layers[-1].output_shape[3]
        self.effectiveKernel,self.effectiveStride=Get_Eff_Kernel_Params(self.intModel)
        
    def Load(self,filename):
        f = open(filename, 'rb')
        tmp_dict = pickle.load(f)
        f.close() 
        
        try:
          self.model=load_model(tmp_dict['hd5'])         
        except:
          self.model=load_model(filename+'.hd5')   
        self.patchSize=tmp_dict['patchSize']
        self.magLevel=tmp_dict['magLevel']
        self.labelNames=tmp_dict['labelNames']

    
        self.intModel=Model(inputs=self.model.input,outputs=self.model.get_layer(index=-2).output)
        self.numberOfClasses=self.intModel.layers[-1].output_shape[3]
        self.effectiveKernel,self.effectiveStride=Get_Eff_Kernel_Params(self.intModel)
        


    def Save(self,fileName,hd5File=None):
        
        if hd5File is None:
            hd5File=fileName+'.hd5'
            
        extraData={'patchSize':self.patchSize,'magLevel':self.magLevel,
                          'labelNames':self.labelNames,'hd5':hd5File}        
        
        pickle.dump(extraData, open(fileName, "wb" ) )
        
        self.model.save(hd5File,save_format='h5')
        

def load_file(im_file):
    return np.array(Image.open(im_file))

        
def Profile_Tumor(classifier,slide,boxHeight=2000,boxWidth=2000, returnScores=True):
   
    patchSize=classifier.patchSize
    magLevel=classifier.magLevel
    stride=classifier.effectiveStride 
    (tumorWidth,tumorHeight)=slide.level_dimensions[magLevel]
   

    nR=int(np.ceil(tumorHeight/boxHeight))
    nC=int(np.ceil(tumorWidth/boxWidth))
    print(nR,nC)
    
    print(int(np.ceil(tumorHeight/stride)),int(np.ceil(tumorWidth/stride)),classifier.numberOfClasses)
    
   
    
    tumorClasses=np.zeros((int(np.ceil(tumorHeight/stride)),int(np.ceil(tumorWidth/stride))),dtype=np.uint8)
    if(returnScores):
      tumorScores=np.zeros((int(np.ceil(tumorHeight/stride)),int(np.ceil(tumorWidth/stride)),classifier.numberOfClasses))
    
    bar=progressbar.ProgressBar(max_value=nR*nC)
    boxCounter=0

    for r in range(nR):
        for c in range(nC):
            imgR=int(r*boxHeight)
            imgC=int(c*boxWidth)
            imgHeight=int(np.minimum(boxHeight,tumorHeight-(r*boxHeight-1)))
            imgWidth=int(np.minimum(boxWidth,tumorWidth-(-1+c*boxWidth)))
        
            x=int((imgC-int(patchSize/2)-1)*slide.level_downsamples[magLevel])
            y=int((imgR-int(patchSize/2)-1)*slide.level_downsamples[magLevel])
            img=np.asarray(slide.read_region((x,y),magLevel,
                                             (imgWidth+patchSize-1,imgHeight+patchSize-1)))[:,:,range(3)]
            outImg=np.squeeze(classifier.intModel.predict(img.reshape((1,img.shape[0],img.shape[1],3))/255))
           
            
            try:
              imgClasses=np.uint8(np.argmax(outImg,axis=2))

            except:
              if(img.shape[0]<img.shape[1]):                
                outImg=outImg.reshape(1,outImg.shape[0],classifier.numberOfClasses)
              else:
                outImg=outImg.reshape(outImg.shape[0],1,classifier.numberOfClasses)
              
              imgClasses=np.uint8(np.argmax(outImg,axis=2))
            
            outHeight=outImg.shape[0]
            outWidth=outImg.shape[1]
            
            
            x=int(np.ceil(imgR/stride))
            y=int(np.ceil(imgC/stride))
            try:
                tumorClasses[x:(x+outHeight),y:(y+outWidth)]=imgClasses
                if returnScores:
                  tumorScores[x:(x+outHeight),y:(y+outWidth),:]=outImg
            except:
                imShape=tumorClasses[x:(x+outHeight),y:(y+outWidth)].shape

                tumorClasses[x:(x+outHeight),y:(y+outWidth)]=imgClasses[0:(imShape[0]),0:(imShape[1])]
                if returnScores:
                  tumorScores[x:(x+outHeight),y:(y+outWidth)]=outImg[0:(imShape[0]),0:(imShape[1]),:]

            boxCounter=boxCounter+1
            bar.update(boxCounter)

        
           
    bar.finish()    
    if returnScores:
      return tumorClasses,tumorScores
    else:
      return tumorClasses


# %%
#NOTE: This class is also implemented in patchgen. Remove one of these.
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
        patchSizeLoaded=np.array(patchSize)
      
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


class SlideAreaGenerator(Sequence):
  """ Uses Keras' generator to create boxes out of the slide. """
  def __init__(self,slide,boxHeight=1000,boxWidth=1000, batch_size=4,borderSize=10, 
                shuffle=False, downSampleFactor=1, preproc_fn = lambda x:np.float32(x)/255):
      """ 
      Initialize the generator to create boxes of the slide. 
      
      ### Returns
      - None
      
      ### Parameters:
      - `slide: slide`  The loaded slide object.
      - `boxHeight: int`  Height of the box that will slide across the slide.
      - `boxWidth: int`  Width of the box that will slide across the slide.
      - `batchSize: int`  Number of boxes to fit in the GPU at time as we profile the slide.
      - `borderSize: int`  Border around the box to profile along with the box itself.
      - `shuffle: bool`  Shuffle indices of boxes in the epoch.
      - `downSampleFactor: int`  Supply reduced size SVS file to profile faster.
      - `preproc_fn: function`  supply preprocessing function. Else, defaults to input/255.
      
      """
      self.slide=slide
      self.boxHeight=boxHeight
      self.boxWidth=boxWidth
      self.batch_size = batch_size
      self.shuffle = shuffle
      
      self.dsf=downSampleFactor
      self.on_epoch_end()
      if self.dsf!=1:
          self.imgReader=PatchReader(slide, downSampleFactor)
          
      (self.slideWidth,self.slideHeight)=slide.dimensions
      self.rVals,self.cVals=np.meshgrid(np.arange(0,self.slideHeight,self.dsf*boxHeight),
                                        np.arange(0,self.slideWidth,self.dsf*boxWidth))          
      self.numberOfBoxes=self.rVals.size
      self.rVals.resize(self.rVals.size)
      self.cVals.resize(self.cVals.size)
      self.borderSize=borderSize
      self.preproc=preproc_fn


  def __len__(self):
      """
      Denotes the number of batches per epoch
      
      ### Returns
      - `int`  Number of batches for the keras generator.
      
      ### Parameters
      - None
      
      """
      return int(np.ceil(self.numberOfBoxes / self.batch_size))

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
      
      indexes=np.arange(index*self.batch_size,np.minimum((index+1)*self.batch_size,self.numberOfBoxes))
    
      
      Y=[self.rVals[indexes],self.cVals[indexes]]
      
      if self.dsf==1:           
          X=np.zeros((len(indexes),self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize,3),dtype=np.float32)   
          for i,idx in enumerate(indexes):
             img=np.zeros((self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize,3))
             r=self.rVals[idx]
             c=self.cVals[idx]
            
             imgHeight=int(np.minimum(self.boxHeight+self.borderSize,self.slideHeight-(r))+self.borderSize)
             imgWidth=int(np.minimum(self.boxWidth+self.borderSize,self.slideWidth-(c))+self.borderSize)
            
             img[0:imgHeight,0:imgWidth]=np.array(self.slide.read_region((c-self.borderSize,r-self.borderSize),0,(imgWidth,imgHeight)))[:,:,range(3)]

             X[i,:,:,:]=self.preproc(img)
      else:
          patchCenters=np.zeros((len(indexes),2))
          
          patchCenters[:,1]=self.rVals[indexes]+self.dsf*int(self.boxHeight/2)
          patchCenters[:,0]=self.cVals[indexes]+self.dsf*int(self.boxWidth/2)
          
          patchSize=(self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize)
          X=self.preproc(self.imgReader.LoadPatches(patchCenters,patchSize))

         
  
          
      return X, Y

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      #self.indexes = np.arange(self.numberOfPatches)

def Profile_Slide_Fast(model, slide, stride, patchSize, numberOfClasses,
                       isMultiOutput=False, downSampleFactor=1, isModelFlattened=True,
                       boxHeight=2000, boxWidth=2000, batchSize=4,
                       useMultiprocessing=True, nWorkers=64, verbose=1,
                       returnActivations=True, preproc_fn = lambda x:np.float32(x)/255):
    """
    Runs the model across the slide and returns the prediction classes and activations of the whole slide.
    
    ### Returns
    - `slidePredictionsList: [np.array(), ...] or np.array()`. List is returned when `isMultiOutput=True`
    - `slideActivationsList: [np.array(), ...] or np.array()`. List is returned when `isMultiOutput=True`
    
    ### Parameters
    - `model: model`  The loaded keras model. Can be intermediate CNN model or FCN.
    - `slide: slide`  The loaded slide object.
    - `stride: int`  The model's stride.
    - `patchSize: int`  The model's patchsize.
    - `numberOfClasses: int`  Total number of classes (including background) the model was trained on.
    - `isMultiOutput: int`  If the model returns one or more outputs for one input.
    - `downSampleFactor: int`  Supply reduced size SVS file to profile faster.
    - `isModelFlattened: int:`  If FCN, mark as True. If intermediate model, then mark False.
    - `boxHeight: int`  Height of the box that will slide across the slide.
    - `boxWidth: int`  Width of the box that will slide across the slide.
    - `batchSize: int`  Number of boxes to fit in the GPU at time as we profile the slide.
    - `useMultiprocessing: bool`  Use the multiprocessing to profile the slide faster.
    - `nWorkers:  int` number of parallel processes to run if useMultiprocessing = True.
    - `verbose: int`  print details as we profile the slides.
    - `returnActivations: bool`  return NN activation outputs along with predicted classes.
    - `preproc_fn: function`  supply preprocessing function. Else, defaults to input/255.
    
    """
    
    if verbose>0:
        start_time = time.time()
    borderSize=int(patchSize/2)
    
    slideGen=SlideAreaGenerator(slide,downSampleFactor=downSampleFactor,
                                boxHeight=boxHeight,boxWidth=boxWidth, 
                                batch_size=batchSize,borderSize=borderSize,
                                preproc_fn=preproc_fn)    

    
    res=model.predict_generator(slideGen,workers=nWorkers,
                                use_multiprocessing=useMultiprocessing,
                                verbose=verbose)
    
    (slideWidth,slideHeight)=slide.dimensions
    outHeight=int(np.ceil((slideHeight-(downSampleFactor*patchSize))/(downSampleFactor*stride)))+1
    outWidth=int(np.ceil((slideWidth-(downSampleFactor*patchSize))/(downSampleFactor*stride)))+1
    
    if not isModelFlattened:
        outBoxHeight=res.shape[1]
        outBoxWidth=res.shape[2]
    else:
        outBoxHeight=int(np.floor(boxHeight/stride))+1
        outBoxWidth=int(np.floor(boxWidth/stride))+1
        if res.size!=res.shape[0]*numberOfClasses*outHeight*outWidth:
            s=int(np.sqrt(res.size/(res.shape[0]*numberOfClasses)))
            outBoxHeight=s
            outBoxWidth=s
            print('Warning, automatic sizing failed, selecting boxsize='+str(outBoxHeight))
            assert res.size==res.shape[0]*numberOfClasses*outBoxHeight*outBoxWidth,'Failed!'
    
    
    
    rVals1,cVals1=np.meshgrid(np.arange(0,slideHeight,boxHeight*downSampleFactor),
                                            np.arange(0,slideWidth,boxWidth*downSampleFactor))
    rVals1.resize(rVals1.size)
    cVals1.resize(cVals1.size)
    
    rVals=np.uint32(np.ceil(rVals1/(downSampleFactor*stride)))
    cVals=np.uint32(np.ceil(cVals1/(downSampleFactor*stride)))
    numberOfBoxes=rVals.size
    
    if isMultiOutput:
        numberOfOutputs=len(model.output)
    else:
        numberOfOutputs=1
   
       
    slideClassesList=[]
    slideActivationsList=[]
    for outNum in range(numberOfOutputs):
        outHeightR=int(outBoxHeight*np.ceil(outHeight/outBoxHeight))
        outWidthR=int(outBoxWidth*np.ceil(outWidth/outBoxWidth))
        slideClasses=np.zeros((outHeightR,outWidthR),dtype=np.uint8)
        if returnActivations:
            slideActivations=np.zeros((outHeightR,outWidthR,numberOfClasses))
        else:
            slideActivations=None
        
        if isMultiOutput:
            response=res[outNum]
        else:
            response=res
            
        
        if isModelFlattened:
            classes=np.argmax(response.reshape(response.shape[0],outBoxHeight,outBoxWidth,numberOfClasses),axis=-1)
            activations=response.reshape(response.shape[0],outBoxHeight,outBoxWidth,numberOfClasses)
        else:
            classes=np.argmax(response,axis=-1)
            activations=response
            
        for i in range(numberOfBoxes):
            r=rVals[i]
            c=cVals[i]
            imgHeight=int(np.minimum(outBoxHeight,outHeightR-(r)))
            imgWidth=int(np.minimum(outBoxWidth,outWidthR-(c)))
            bS=0
            slideClasses[r:(r+imgHeight),c:(c+imgWidth)]=classes[i][bS:(bS+imgHeight),bS:(bS+imgWidth)]
            
            if returnActivations:
                slideActivations[r:(r+imgHeight),c:(c+imgWidth)]=activations[i][bS:(bS+imgHeight),bS:(bS+imgWidth)]
                
        
        if isMultiOutput:
            slideClassesList.append(slideClasses[0:outHeight,0:outWidth])
            if returnActivations:
                slideActivationsList.append(slideActivations[0:outHeight,0:outWidth,:])
        else:
            slideClassesList=slideClasses[0:outHeight,0:outWidth]
            if returnActivations:
                slideActivationsList=slideActivations[0:outHeight,0:outWidth,:]            
  
        

    
    if verbose>0:        
        print("--- %s seconds ---" % (time.time() - start_time))        
    return slideClassesList,slideActivationsList

