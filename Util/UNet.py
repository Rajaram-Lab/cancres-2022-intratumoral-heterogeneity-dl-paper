"""
This file contains functions and custom objects for the usage of Unet in the project. 

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
ROOT_DIR=wrkDir

sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.join(ROOT_DIR, 'Util'))

import numpy as np

import tensorflow as tf

from tensorflow.keras.layers import Conv2D, MaxPooling2D,Input,UpSampling2D,Lambda
from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from functools import partial
import time

# %%
def image_softmax(input):
  label_dim = -1
  d = K.exp(input - K.max(input, axis=label_dim, keepdims=True))
  return d / K.sum(d, axis=label_dim, keepdims=True)
image_softmax.__name__='image_softmax'


def crop_and_concatenate(comb):
    x1=comb[0]
    x2=comb[1]
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [x1_shape[0], x2_shape[1], x2_shape[2], x1_shape[3]]
    x1_crop=x1[:,offsets[1]:(offsets[1]+size[1]),offsets[2]:(offsets[2]+size[2]),:]
    return K.concatenate([x1_crop, x2], -1)


def concat_output_shape(comb):
    shape=list(tf.shape(comb[0]))
    shape[3]=shape[3]+tf.shape(comb[1])
    return tuple(shape)


def zeropad(comb):
    inputTensor=comb[0]
    targetTensor=comb[1]
    inputshape = tf.shape(inputTensor)
    targetshape = tf.shape(targetTensor)
    paddings=[[0,0],[0,targetshape[1]-inputshape[1]],[0,targetshape[2]-inputshape[2]],[0,0]]
    # offsets for the top left corner of the crop
    return tf.pad(inputTensor,paddings,'CONSTANT',constant_values=0)


__EPS = 1e-5

def weighted_image_categorical_crossentropy(y_true, y_pred,weights):
    y_pred = K.clip(y_pred, __EPS, 1 - __EPS)
    score=-K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred),axis=-1)
    w1=K.reshape(K.gather(weights,K.argmax(y_true)),K.shape(score))

    return K.mean(score*w1)

class UNet():
    def __init__(self):
        print ('build UNet ...')

    def get_crop_shape(self, target, refer):
        # width, the 3rd dimension
        cw = (target.get_shape()[2] - refer.get_shape()[2]).value
        assert (cw >= 0)
        if cw % 2 != 0:
            cw1, cw2 = int(cw/2), int(cw/2) + 1
        else:
            cw1, cw2 = int(cw/2), int(cw/2)
        # height, the 2nd dimension
        ch = (target.get_shape()[1] - refer.get_shape()[1]).value
        assert (ch >= 0)
        if ch % 2 != 0:
            ch1, ch2 = int(ch/2), int(ch/2) + 1
        else:
            ch1, ch2 = int(ch/2), int(ch/2)

        return (ch1, ch2), (cw1, cw2)
    

    def create_model(self, img_shape, num_class):
        inputs = Input(shape = img_shape)
        
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', name='conv1_1')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up_conv5 = UpSampling2D(size=(2, 2))(conv5)
       
        up6=Lambda(crop_and_concatenate,output_shape=concat_output_shape)([conv4,up_conv5])
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up_conv6 = UpSampling2D(size=(2, 2))(conv6)
       
        up7=Lambda(crop_and_concatenate)([conv3,up_conv6])
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up_conv7 = UpSampling2D(size=(2, 2))(conv7)
        
        up8=Lambda(crop_and_concatenate)([conv2,up_conv7])
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up_conv8 = UpSampling2D(size=(2, 2))(conv8)
        
        up9=Lambda(crop_and_concatenate)([conv1,up_conv8])
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv9=Lambda(zeropad)([conv9,inputs])
        conv10 = Conv2D(num_class, (1, 1),activation=image_softmax)(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model


# %%

class SlideAreaGenerator(Sequence):
  def __init__(self,slide,boxHeight=1000,boxWidth=1000, batch_size=4,borderSize=10, 
                shuffle=False):
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
      (self.slideWidth,self.slideHeight)=slide.dimensions
      self.rVals,self.cVals=np.meshgrid(np.arange(0,self.slideHeight,boxHeight),np.arange(0,self.slideWidth,boxWidth))          
      self.numberOfBoxes=self.rVals.size
      self.rVals.resize(self.rVals.size)
      self.cVals.resize(self.cVals.size)
      self.borderSize=borderSize
      self.on_epoch_end()

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
    
      X=np.zeros((len(indexes),self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize,3),dtype=np.float32)
      Y=[self.rVals[indexes],self.cVals[indexes]]
      for i,idx in enumerate(indexes):
         img=np.zeros((self.boxHeight+2*self.borderSize,self.boxWidth+2*self.borderSize,3))
         r=self.rVals[idx]
         c=self.cVals[idx]
         imgHeight=int(np.minimum(self.boxHeight+self.borderSize,self.slideHeight-(r))+self.borderSize)
         imgWidth=int(np.minimum(self.boxWidth+self.borderSize,self.slideWidth-(c))+self.borderSize)
         
         img[0:imgHeight,0:imgWidth]=np.array(self.slide.read_region((c-self.borderSize,r-self.borderSize),0,(imgWidth,imgHeight)))[:,:,range(3)]/255

         X[i,:,:,:]=img

         
  
          
      return X, Y

  def on_epoch_end(self):
      """ Updates indexes after each epoch """
      #self.indexes = np.arange(self.numberOfPatches)

# %%
def Profile_Slide_Fast(model,slide,boxHeight=1000,boxWidth=1000, batchSize=4,borderSize=10,
                       useMultiprocessing=True,nWorkers=64,verbose=1,responseThreshold=None,bgClass=0):
    """
    Runs the model across the slide and returns the prediction classes and activations of the whole slide.
    
    ### Returns
    - `slidePredictions:  np.array()`  numpy array of predictions after running the model across the slide.
    
    ### Parameters
    - `slidePredictionsList: np.array()`.  Predicted numpy array of whole slide. Dimensions of output is going to be the same as the slide.
    - `slide: slide`  The loaded slide object.
    - `boxHeight: int`  Height of the box that will slide across the slide.
    - `boxWidth: int`  Width of the box that will slide across the slide.
    - `batchSize: int`  Number of boxes to fit in the GPU at time as we profile the slide.
    - `useMultiprocessing: bool`  Use the multiprocessing to profile the slide faster.
    - `nWorkers:  int` number of parallel processes to run if useMultiprocessing = True.
    - `verbose: int`  print details as we profile the slides.
    - `responseThreshold: float or None`  Set the threshold under which predictions must be called background.
    - `bgClass: int`  The index for bgClass in the model.
    
    """
    
    slideGen=SlideAreaGenerator(slide,boxHeight=boxHeight,boxWidth=boxWidth, batch_size=batchSize,borderSize=borderSize)    
    if verbose>0:
        start_time = time.time()
    
    res=model.predict_generator(slideGen,workers=nWorkers,use_multiprocessing=useMultiprocessing,verbose=verbose)

    classes=np.argmax(res,axis=-1)
    if responseThreshold is not None:
        maxRes=np.max(res,axis=-1)
        classes[maxRes<responseThreshold]=bgClass


    (slideWidth,slideHeight)=slide.dimensions
    rVals,cVals=np.meshgrid(np.arange(0,slideHeight,boxHeight),np.arange(0,slideWidth,boxWidth))          
    numberOfBoxes=rVals.size
    rVals.resize(rVals.size)
    cVals.resize(cVals.size)
    slideClasses=np.zeros((slideHeight,slideWidth),dtype=np.uint8)
    for i in range(numberOfBoxes):
        r=rVals[i]
        c=cVals[i]
        imgHeight=int(np.minimum(boxHeight,slideHeight-(r)))
        imgWidth=int(np.minimum(boxWidth,slideWidth-(c)))
        slideClasses[r:(r+imgHeight),c:(c+imgWidth)]=classes[i][borderSize:(borderSize+imgHeight),borderSize:(borderSize+imgWidth)]
    if verbose>0:        
        print("--- %s seconds ---" % (time.time() - start_time))        
    return slideClasses
# %% meanIoU ignoring background(0) and edge(2) pixels

def MeanIOU(y_true, y_pred):
    # DO NOT USE WITHOUT GENERALIZING 
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels_BG = K.equal(K.sum(y_true, axis=-1), 0)
    void_labels_Edge = K.equal(K.sum(y_true, axis=-1), 2)
    for i in range(0, nb_classes): # exclude first label (background) and last label (void)
        # '& ~void_labels_Edge'  was added to ignore Edge pixels along with BG pixels
        true_labels = K.equal(true_pixels, i) & ~void_labels_BG & ~void_labels_Edge
        pred_labels = K.equal(pred_pixels, i) & ~void_labels_BG & ~void_labels_Edge
        inter = tf.cast(true_labels & pred_labels, tf.int32)
        union = tf.cast(true_labels | pred_labels, tf.int32)
        legal_batches = K.sum(tf.cast(true_labels, dtype=tf.int32), axis=1) > 0
        ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        iou.append(K.mean(ious[legal_batches]))
    iou = tf.stack(iou)
    legal_labels = ~tf.math.is_nan(iou)
    iou = iou[legal_labels]
    return K.mean(iou)
MeanIOU.__name__='MeanIOU'

# %%

def LoadUNet(modelFile, custom_objects={}):    
    w=np.random.rand(2)
    w=np.float32(np.max(w)/w)
    myLoss = partial(weighted_image_categorical_crossentropy,weights=w)  
    myLoss.__name__='weighted_loss'
    all_custom_objects={'image_softmax':image_softmax, 'weighted_loss':myLoss,'tf':tf,'MeanIOU':MeanIOU}
    # add new 
    for key in custom_objects:
        all_custom_objects[key] = custom_objects[key]
    uNetModel=load_model(modelFile,custom_objects=all_custom_objects)
    return uNetModel
