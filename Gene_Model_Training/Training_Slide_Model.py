
"""
Train_Slide_Level is used to train all gene models used for slide level classification.
The model file names are found in the respective parameter file (Model_Files.yaml)
    
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


from tensorflow.keras.callbacks import ModelCheckpoint
import progressbar

import Util.PatchGen as pg
import Util.extendedImgaugAugmenters as exiaa
import Util.MILCore as mil

with open(os.path.join(ROOT_DIR, 'Parameters/Training_Params.yaml')) as file:
    trainParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR, 'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
with open(os.path.join(ROOT_DIR, 'Parameters/Model_Files.yaml')) as file:
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
    parser = argparse.ArgumentParser(description='Train a Attention-based Deep MIL')
    
    parser.add_argument('--geneToAnalyze', dest='geneToAnalyze',
                        help='Specify which gene to analyze :: BAP1, PBRM1, SETD2',
                        default='BAP1', type=str)
    parser.add_argument('--foldNum', dest='foldNum',
                    help='What fold to use as validation fold :: 0,1,2',
                    default=0, type=int)
 
    args = parser.parse_args()
    return args

# %% extract parameters from cmd line arguments
args = parse_args()

# check arg, argValue, argType
for arg in vars(args):
     print(arg, getattr(args, arg), type(getattr(args, arg)))

# extract parameters from pickle file
max_epoch = trainParams['MIL']['max_epoch']
magLevel = trainParams['MIL']['magLevel']
batchSize = trainParams['MIL']['batchSize']
input_dim = trainParams['MIL']['input_dim']

allSampleFile = os.path.join(ROOT_DIR, projectPaths['Data']['AllSamples'])
outputPatchDir = os.path.join(dataDir, projectPaths['Data']['PatchData'],
                              'WSI/')
foldsIdx = os.path.join(ROOT_DIR, projectPaths['Data']['FoldsIdx'])


#%% Define model paths
modelType = 'MIL_Models'
modelDir = os.path.join(dataDir, modelFiles[modelType]['BaseDir'])

checkpointPath = os.path.join(modelDir, f"{args.geneToAnalyze}_{str(args.foldNum)}F_Slide-" + "{epoch:02d}.hdf5")

# %%
def load_dataset(geneToAnalyze, foldNum):
    # load fold indices' pickle file and split them into train and validation folds
    with open(os.path.join(ROOT_DIR, foldsIdx), 'rb') as fh:
        foldIndices = pickle.load(fh)
    
    folds={}
    folds[0]={'train':np.concatenate((foldIndices[1],foldIndices[2])),'test':foldIndices[0]}
    folds[1]={'train':np.concatenate((foldIndices[0],foldIndices[2])),'test':foldIndices[1]}
    folds[2]={'train':np.concatenate((foldIndices[0],foldIndices[1])),'test':foldIndices[2]}
    folds['holdout']={'train':[],'test':foldIndices[3]} # never used throughout the repo
    
    trainIdx=folds[foldNum]['train']
    testIdx=folds[foldNum]['test']
    
    # read table indicating the 
    allSamples = pd.read_csv(os.path.join(ROOT_DIR, allSampleFile)).drop(['Unnamed: 0'], axis=1)
    
    trainSamples=allSamples.iloc[trainIdx]
    trainHdf5List=[os.path.join(outputPatchDir, f.replace('.svs','.hdf5')) for f in trainSamples.svs.values]
    # boolean array to drop any hdf5 files and corresponding rows whose patch files are not available.
    isPresent=np.array([os.path.exists(f) for f in trainHdf5List])
    trainHdf5List=np.array(trainHdf5List)[isPresent].tolist()
    # the class assignment for this sample's patches
    trainSampleToClass=np.uint8(trainSamples[geneToAnalyze+ '_Positive'].values[isPresent])
    # indicate if this sample is a focal case or not
    trainSampleToFocal=np.uint8(trainSamples[geneToAnalyze+ '_Focal'].values[isPresent])
    # load patches from patchFiles list
    trainPatchData,temp,temp1,trainSampleNumbers=pg.LoadPatchData(trainHdf5List, returnSampleNumbers=True)
    # get the class of the patch by tracking back the sample they come from
    trainPatchClasses=trainSampleToClass[np.int32(trainSampleNumbers)]
    # arrange them in a tuple
    trainData=[trainPatchData[0],np.array(trainPatchClasses),np.array(trainSampleNumbers),np.array(trainSampleToFocal)]

    testSamples=allSamples.iloc[testIdx]
    testHdf5List=[os.path.join(outputPatchDir,f.replace('.svs','.hdf5')) for f in testSamples.svs.values]
    # boolean array to drop any hdf5 files and corresponding rows whose patch files are not available.
    isPresent=np.array([os.path.exists(f) for f in testHdf5List])
    testHdf5List=np.array(testHdf5List)[isPresent].tolist()
    # the class assignment to this sample's patches
    testSampleToClass=np.uint8(testSamples[geneToAnalyze+ '_Positive'].values[isPresent])
    # indicate if this sample is a focal case or not
    testSampleToFocal=np.uint8(testSamples[geneToAnalyze+ '_Focal'].values[isPresent])
    # load patches from patchFiles list
    testPatchData,temp,temp1,testSampleNumbers=pg.LoadPatchData(testHdf5List, returnSampleNumbers=True)
    # get the class of the patch by tracking back the sample they come from
    testPatchClasses=testSampleToClass[np.int32(testSampleNumbers)]
    # arrange them in a tuple
    testData=[testPatchData[0],np.array(testPatchClasses),np.array(testSampleNumbers),np.array(testSampleToFocal)]
    
    return trainData, testData

#%% Load dataset

train_set, test_set = load_dataset(args.geneToAnalyze, args.foldNum)

# %%

augVals=trainParams['augmentations']
augmentations=exiaa.HEDadjust(
    hAlpha = augVals['hAlpha'],
    eAlpha = augVals['eAlpha'],
    rAlpha = augVals['rAlpha'])


trainGen = mil.MILSlideGen(data=train_set[0], labels=train_set[1], 
                            sampleNumbers=train_set[2],
                            numberOfClasses=2, batch_size=batchSize,
                            categoricalLabels=False, augmentations=augmentations)

testGen = mil.MILSlideGen(data=test_set[0], labels=test_set[1], 
                            sampleNumbers=test_set[2],
                            numberOfClasses=2, batch_size=batchSize,
                            categoricalLabels=False)

classNumbers=[]
print('Calculating class weights')
bar=progressbar.ProgressBar(max_value=2000)
for i in range(2000): # Meant to get a large number of batches so we get a reliable readout of class composition
    _, classes = trainGen[i]
   
    batchClass=classes[0]
    classNumbers.append(batchClass)
    bar.update(i)
bar.finish()

classNumbers=np.array(classNumbers)
classCounts=np.array([np.sum(classNumbers==0),np.sum(classNumbers==1)])
class_weights=(np.max(classCounts)/classCounts).astype(np.float32)

#%% Define model

model=mil.MilNetwork(input_dim,trainParams,class_weights)

# %% Define Checkpoints and train model
checkpoint = ModelCheckpoint(checkpointPath, monitor='val_categorical_accuracy', 
                             verbose=0, save_best_only=False, save_weights_only=False, 
                             mode='auto', save_freq='epoch')
callbacks = [checkpoint]


history = model.fit_generator(generator=trainGen, 
                              steps_per_epoch=np.floor(trainGen.numberOfPatches/trainGen.batch_size),
                              epochs=max_epoch, 
                              validation_data=testGen,
                              callbacks=callbacks,use_multiprocessing=True)
