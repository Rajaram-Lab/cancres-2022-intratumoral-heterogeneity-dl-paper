"""
Train_Region_Model is used to train all gene models used in region classification.
Training parameters and model file names are found in the respective parameter 
files (Training_Params and Model_Files.yaml).

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

import os,sys
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir
sys.path.insert(0,ROOT_DIR)
CUDA_VISIBLE_DEVICES=0

import yaml, pickle, argparse

import numpy as np
import pandas as pd

from collections import Counter
from tensorflow.keras.layers import Conv2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import SGD

import Util.PatchGen as pg
import Util.extendedImgaugAugmenters as exiaa

with open(os.path.join(ROOT_DIR, 'Parameters/Training_Params.yaml')) as file:
    trainParams = yaml.full_load(file)

with open(os.path.join(ROOT_DIR, 'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)

with open(os.path.join(ROOT_DIR,'Parameters/Model_Files.yaml')) as file:
    modelFiles = yaml.full_load(file)

dataDir=projectPaths['DataDir']
#%% Define args
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
    parser = argparse.ArgumentParser(description='Automated job submission')
    
    parser.add_argument('--geneToAnalyze', dest='geneToAnalyze',
                        help='Specify which gene to analyze :: BAP1, PBRM1, SETD2',
                        default='BAP1', type=str)
    parser.add_argument('--foldNum', dest='foldNum',
                        help='Which fold to use as validation :: 0,1,2',
                        default=2, type=int)

    args = parser.parse_args()
    return args

args=parse_args()

print('Called with args:')
print(args)

#%%Define all variables 
geneToAnalyze=args.geneToAnalyze
foldNum=args.foldNum

magLevel=trainParams['Region']['magLevel']
numberOfEpochs=trainParams['Region']['numEpochs']
patchSize=trainParams['Region']['patchSize']
numberOfClasses=trainParams['Region']['numberOfClasses']

outputPatchDir=os.path.join(dataDir,projectPaths['Data']['PatchData'],"WSI/")
focalPatchDir=os.path.join(dataDir,projectPaths['Data']['FocalPatches'])

allSampleFile=os.path.join(ROOT_DIR,projectPaths['Data']['AllSamples'])
foldsIdx=os.path.join(ROOT_DIR,projectPaths['Data']['FoldsIdx'])

#%%
# Print parameters to know what the job is running 
print("geneToAnalyze: ",geneToAnalyze)
print("numberOfEpochs: ", numberOfEpochs)
print("magLevel: ", magLevel)
print("foldsIdx: ", foldsIdx)
print("foldNum: ", foldNum)
print("patchSize: ", patchSize)
print("numberOfClasses: ", numberOfClasses)
print("focalPatchDir: ", focalPatchDir)
print("outputPatchDir: ", outputPatchDir)

# %% Define model paths
modelType='FCN_Models'

modelDir=os.path.join(dataDir,
                      modelFiles[modelType]['BaseDir'])
checkpointPath=os.path.join(modelDir,geneToAnalyze+"_"+str(foldNum)+"F_Region-{epoch:02d}.hdf5")

# %% Load the training and testing data into memory


def load_dataset(geneToAnalyze, magLevel, foldNum):
    if not magLevel == '20X':
        raise Exception("Magnification should be 20X")
    
    with open(foldsIdx, 'rb') as f:
        folds = pickle.load(f)
    testIdx=folds[foldNum] # changed from validIdx to testIdx since rest of the this function says test everywhere
    # should be called validation throughout actually
    
    # numRows in trainIdx = 517
    # numRows in testIdx = 265 (i.e. validationIdx)
    # numRows in holdOutIdx = 1292-(517+265) = 510
    
    if foldNum == 0: trainIdx=np.array(list(folds[1])+list(folds[2]))
    elif foldNum == 1: trainIdx=np.array(list(folds[0])+list(folds[2]))
    elif foldNum == 2: trainIdx=np.array(list(folds[0])+list(folds[1]))
     
    # Load the training and testing data into memory
    allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)
    
    # extract training rows, extract training-set's nonFocal samples and their corresponding patch files (hdf5 extension)
    trainSamples=allSamples.iloc[trainIdx]
    trainNonFocalSamples=trainSamples.iloc[np.where(trainSamples[geneToAnalyze+'_Focal'].values==False)[0]]
    trainHdf5ListNF=[os.path.join(outputPatchDir,f.replace('.svs','.hdf5')) for f in trainNonFocalSamples.svs.values]
    
    # extract testing rows, extract testing-set's nonFocal samples and their corresponding patch files (hdf5 extension)
    testSamples=allSamples.iloc[testIdx]
    testNonFocalSamples=testSamples.iloc[np.where(testSamples[geneToAnalyze+"_Focal"].values==False)[0]]
    testHdf5ListNF=[os.path.join(outputPatchDir,f.replace('.svs','.hdf5')) for f in testNonFocalSamples.svs.values]
    
    # get sample numbers that is gene-positive (1) and gene-negative(0), load patches using hdf5 files and return the sample numbers
    # use the returned sample numbers to assign classes to the patches. All patches from a single sample are assigned the same class.
    trainSampleToClassNF=np.uint8(trainNonFocalSamples[geneToAnalyze+ '_Positive'].values)
    trainPatchDataNF,temp,temp1,trainSampleNumbersNF=pg.LoadPatchData(trainHdf5ListNF,returnSampleNumbers=True)    
    trainPatchClassesNF=trainSampleToClassNF[np.int32(trainSampleNumbersNF)]
    # returned array will be of same size as sampleNumber array. Will contain the class for each patch
   
   # same operations on the test set
    testSampleToClassNF=np.uint8(testNonFocalSamples[geneToAnalyze+ '_Positive'].values)
    testPatchDataNF,temp,temp1,testSampleNumbersNF=pg.LoadPatchData(testHdf5ListNF,returnSampleNumbers=True)    
    testPatchClassesNF=testSampleToClassNF[np.int32(testSampleNumbersNF)]
    
    # incase of BAP1, we have data on both Focal and nonFocal. We will be including both types of samples in our training/testing data.
    # incase of PBRM1, SETD2, we will include only nonFocal cases. 
    if geneToAnalyze=='BAP1': # append Focal cases to the train and test data.
        trainFocalSamples=trainSamples.iloc[np.where(trainSamples[geneToAnalyze+"_Focal"].values==True)[0]]
        trainHdf5ListF=[os.path.join(focalPatchDir,f.replace('.svs','.hdf5')) for f in trainFocalSamples.svs.values]
        trainPatchDataF,trainPatchClassesF,temp1,temp2=pg.LoadPatchData(trainHdf5ListF,returnSampleNumbers=True)      
        
        testFocalSamples=testSamples.iloc[np.where(testSamples[geneToAnalyze+"_Focal"].values==True)[0]]
        testHdf5ListF=[os.path.join(focalPatchDir,f.replace('.svs','.hdf5')) for f in testFocalSamples.svs.values]
        testPatchDataF,testPatchClassesF,temp1,temp2=pg.LoadPatchData(testHdf5ListF,returnSampleNumbers=True)
        
        # append labels for train and test data
        trainPatchClasses=list(trainPatchClassesNF)+list(trainPatchClassesF)
        testPatchClasses=list(testPatchClassesNF)+list(testPatchClassesF)
       
       # append input data for train and test data
        trainPatchData=np.concatenate((trainPatchDataNF[0],trainPatchDataF[0]),axis=0)
        trainData=[trainPatchData,np.array(trainPatchClasses)]          
        testPatchData=np.concatenate((testPatchDataNF[0],testPatchDataF[0]),axis=0)
        testData=[testPatchData,np.array(testPatchClasses)]
    else:
        trainPatchData=trainPatchDataNF[0]
        trainData=[trainPatchData,np.array(trainPatchClassesNF),np.array(trainSampleNumbersNF)]                      
        testPatchData=testPatchDataNF[0]
        testData=[testPatchData,np.array(testPatchClassesNF)] 
    
    return trainData, testData

trainData,testData= load_dataset(geneToAnalyze,magLevel,foldNum)

#%% Custom Data Generator
  
class CustomDataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, data, labels, numberOfClasses, batch_size=128,
                shuffle=True,preproc_fn= lambda x:np.float32(x)/255,
                augmentations=None):
      'Initialization'
      self.data=data
      self.numberOfPatches=data.shape[0]
      self.labels = labels
      self.numberOfClasses=numberOfClasses
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.augmentations = augmentations
      self.preproc=preproc_fn
      self.on_epoch_end()
 
 
  def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(self.numberOfPatches / self.batch_size))
  
  
  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
      X = np.zeros((len(indexes), self.data.shape[1], self.data.shape[2], self.data.shape[3]),dtype=np.float32)
      
      for i,idx in enumerate(indexes):
         currPatch = np.squeeze(self.data[idx, :, :, :])
         if self.augmentations is not None:
             currPatch = self.augmentations.augment_image(currPatch)
         X[i,:,:,:]=self.preproc(currPatch)

      return X, to_categorical(self.labels[indexes], num_classes=self.numberOfClasses)

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(self.numberOfPatches)
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
          pass

#%% Define Model

def LoadModel():
    # make use of pretrained weights for our model
    model = VGG19(weights = "imagenet", include_top=False, input_shape = (None, None, 3)) 
    # Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
    for layer in model.layers[:5]:
        layer.trainable = False
    
    #Adding custom Layers 
    x = model.output
    x = Conv2D(1024,[7,7],activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024,[1,1],activation='relu')(x)
    x = Conv2D(2,[1,1],activation='softmax')(x)
    predictions = Flatten()(x)
    myModel = Model(inputs= model.input, outputs = predictions)
    return myModel

#%% Augmentations 

augVals=trainParams['augmentations']
augmentations=exiaa.HEDadjust(
    hAlpha = augVals['hAlpha'],
    eAlpha = augVals['eAlpha'],
    rAlpha = augVals['rAlpha'])


#%% Define Generators
trainGen=CustomDataGenerator(trainData[0],trainData[1],numberOfClasses,
                             augmentations=augmentations)
testGen=CustomDataGenerator(testData[0],testData[1],numberOfClasses)


#%% Train Model

import time

start = time.time()
print("Start timer")

checkpoint = ModelCheckpoint(checkpointPath, monitor='val_categorical_accuracy', verbose=0,
                             save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch')
cb_list = [checkpoint]

lr=trainParams['Region']['lr']
momentum=trainParams['Region']['momentum']

myModel=LoadModel()
myModel.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=lr,momentum=momentum),
              metrics=['categorical_accuracy'])

counter = Counter(trainGen.labels)                          
max_val = float(max(counter.values()))       
class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}  
history=myModel.fit_generator(generator=trainGen,
                              steps_per_epoch=np.floor(trainGen.numberOfPatches/trainGen.batch_size),
                              epochs=numberOfEpochs, 
                              class_weight=class_weights,
                              validation_data=testGen,
                              callbacks=cb_list,
                              validation_steps=10)

end = time.time()
print(end - start)
