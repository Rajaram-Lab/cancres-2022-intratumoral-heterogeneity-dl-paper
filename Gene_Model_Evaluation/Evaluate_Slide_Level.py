"""
Evaluate_Slide_Level is used to evaluate all slide gene models across all testing cohorts.
The model file names are found in the respective parameter file (Parameters/Model_Files.yaml)

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
CUDA_VISIBLE_DEVICES=0

import yaml


with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Model_Files.yaml')) as file:
    modelFiles = yaml.full_load(file)

import Util.MILCore as mil
import Util.PatchGen as pg
import numpy as np
import pandas as pd
import pickle
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.utils import to_categorical
import progressbar
import argparse

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
        evaluate all slide gene models on specified cohorts.')
    parser.add_argument('--gene', dest='gene',
                        help='specify gene to evaluate',
                        default='BAP1', type=str)
    parser.add_argument('--cohort', dest='cohort',
                        help='TCGA, WSI',
                        default='TCGA', type=str)
    args = parser.parse_args()
    return args

args=parse_args()

dataDir=projectPaths['DataDir']
# %%
class SimpleDataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, data, labels, numberOfClasses, batch_size=128, 
                shuffle=True,preproc_fn= lambda x:np.float32(x)/255):
      'Initialization'
      self.data=data
      self.numberOfPatches=data.shape[0]
      self.labels = labels       
      self.numberOfClasses=numberOfClasses
      self.batch_size = batch_size
      self.shuffle = shuffle
      self.preproc=preproc_fn
      
           
      self.on_epoch_end()

  def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(self.numberOfPatches / self.batch_size))

  def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
      
      X=np.zeros((len(indexes),self.data.shape[1],self.data.shape[2],self.data.shape[3]),dtype=np.float32)
      
      for i,idx in enumerate(indexes):
         X[i,:,:,:]=self.preproc(np.squeeze(self.data[idx,:,:,:]))

      return X, to_categorical(self.labels[indexes], num_classes=self.numberOfClasses)

  def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(self.numberOfPatches)
      if self.shuffle == True:
          np.random.shuffle(self.indexes)

# %%

geneToAnalyze=args.gene # BAP1,PBRM1,SETD2
cohort=args.cohort # WSI or TCGA

assert cohort in ['WSI','TCGA'], "Slide level evaluation is only performed on TCGA and WSI"
for foldNum in [0,1,2]:
    if cohort =='WSI':
        h5Dir=os.path.join(dataDir,projectPaths['Data']['PatchData'],'WSI/')
        
        sampleInfoFile=os.path.join(ROOT_DIR,projectPaths['Data']['AllSamples'])
        
        foldsFile=os.path.join(ROOT_DIR, projectPaths['Data']['FoldsIdx'])
        
        foldIndices=pickle.load(open(foldsFile,'rb'))
        
        holdoutIdx=foldIndices[3]
           
        allSamples = pd.read_csv(sampleInfoFile).drop(['Unnamed: 0'], axis=1)
           
        holdoutSamples=allSamples.iloc[holdoutIdx]
        allH5List=[os.path.join(h5Dir,f.replace('.svs','.hdf5')) for f in holdoutSamples['svs'].values]
        isGood=np.isfinite(holdoutSamples[geneToAnalyze+'_Positive'].values.astype(np.float32))
        isPresent=np.array([os.path.exists(f) for f in allH5List])
        goodSamples=holdoutSamples[np.logical_and(isGood,isPresent)]
           
        allH5List=[os.path.join(h5Dir,f.replace('.svs','.hdf5')) for f in goodSamples['svs'].values]
        
        allIsFocal=goodSamples[geneToAnalyze+'_Focal'].values
        allIsWT=np.logical_and(goodSamples[geneToAnalyze+'_Positive'].values, np.logical_not(allIsFocal))
        nSamples=len(allH5List)
        
        saveDir=os.path.join(dataDir,projectPaths['ResponseData'],"Slide_Level",'WSI/')
        
        saveFile=os.path.join(saveDir,geneToAnalyze+'_'+cohort+'_Fold'+str(foldNum)+'.pkl')  
            
    elif cohort=='TCGA':
       
        h5Dir=os.path.join(dataDir, projectPaths['Data']['PatchData'],'TCGA/')
        
        geneData=pd.read_csv(os.path.join(ROOT_DIR,
                                          projectPaths['Slide_Level']['Genetics'])).drop(['Unnamed: 0'], axis=1)
        
        geneDataNames=np.array([x.rsplit('.',1)[0] for x in geneData.svs.values])
        allH5List=[os.path.join(h5Dir, x+'.hdf5') for x in geneDataNames]
        allIsWT=geneData['is'+geneToAnalyze+'WT_True'].values
       
       
        allIsFocal=np.zeros(len(allH5List),dtype=bool)
        nSamples=len(allH5List)    
       
        saveDir=os.path.join(dataDir,projectPaths['ResponseData'],"Slide_Level","TCGA/")
        
        saveFile=os.path.join(saveDir,geneToAnalyze+'_'+cohort+'_Fold'+str(foldNum)+'.pkl')  
        
        
    milModelDir= os.path.join(dataDir,modelFiles['MIL_Models']['BaseDir'])

    foldToModel=modelFiles['MIL_Models'][geneToAnalyze]
                              
    milModelFile=os.path.join(milModelDir,foldToModel[foldNum])
    milModel=load_model(milModelFile,compile=False,custom_objects=\
                        {'Normalize_Layer':mil.Normalize_Layer,
                         'Last_Sigmoid':mil.Last_Sigmoid})
    
    classDict={'neg':0,'pos':1}
    allActivations=[]
    bar=progressbar.ProgressBar(max_value=nSamples)
    
    if not os.path.exists(saveFile):
        for i in range(nSamples):
            try:
                h5Data=pg.LoadHdf5Data(allH5List[i])
                classLabels=np.zeros(len(h5Data['classLabels']))
                patchData=h5Data['patchData'][0]['patches']
                dataGen=SimpleDataGenerator(patchData, classLabels,2)
                allActivations.append(milModel.predict_generator(dataGen))
            except ValueError:  #-- Error is called if len(patches) < batchSize
                print('\nNot enough patches for batch', allH5List[i])
                allActivations.append(np.array([]))
            except IndexError:  #-- Error called if patches = []
                print('\nNo patches found for file', allH5List[i])
                allActivations.append(np.array([]))
            bar.update(i)
        bar.finish()
        
        pickle.dump([allIsWT,allIsFocal,allActivations,allH5List,milModelFile],
                    open(saveFile, "wb" ) )
        
    else: 
        print(saveFile,' Already Exists!')
    
    
