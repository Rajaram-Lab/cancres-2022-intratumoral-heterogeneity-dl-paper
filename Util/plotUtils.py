#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plotUtils contains all utility functions that are used to load result data, 
perform analysis, and plot figures. It is directly called by all figure generation
files.

Note: Any instances of the code that use the term Focal are referring to the 
Localized loss status described in the paper. "Focal" or "isFocal=True" indicates
that the sample is a localized loss sample.

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

import openslide as oSlide
import Util.PatchGen as pg 
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


from matplotlib.colors import ListedColormap

import pandas as pd 
import cv2
from skimage.transform import resize
import yaml
from tqdm import tqdm
from skimage.measure import block_reduce

with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Training_Params.yaml')) as file:
    trainParams = yaml.full_load(file)
with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
  

#%%
def ROC_Plot(truePosLabels,predictedPctPositive,color,geneName,ls='-',
             project=None, plot=True, title=None):
    """
    This function is used to perform all AUC_ROC analyses and plot the corresponding 
    plots.

    Parameters
    ----------
    truePosLabels : array
        Ground truth labels for the data.
    predictedPctPositive : array
        Continuous prediction scores for the data.
    color : string
        This defines the color of the ROC_AUC plot. Must be an input acceptable
        by matplotlib colors.
    geneName : string
        This is the name of the gene being analyzed. It is used only for legend 
        names.
    ls : string, optional
        Defines the linestyle of the ROC_AUC curve. The default is '-'.
    project : string, optional
        Defines the cohort being analyzed (TMA1, TMA2, PDX1). Only used for legend.
        The default is None.
    plot : bool, optional
        Plot ROC_AUC curve (True) or only return the roc_auc, fpr, and tpr. 
        The default is True.
    defaultTitle : bool, optional
        Plot ROC_AUC curve (True) or only return the roc_auc, fpr, and tpr. 
        The default is True.
    Returns
    -------
    roc_auc : float
        Area under the curve.
    fpr : float
        False positive rate.
    tpr : float
        True positive rate.

    """
    fpr, tpr, _ = roc_curve(1-truePosLabels[~np.isnan(predictedPctPositive)],
                                            1-predictedPctPositive[~np.isnan(predictedPctPositive)])
    roc_auc = auc(fpr, tpr)
    if plot:
       
        if project==None:
            plt.plot(fpr, tpr, color=color,linewidth=3,ls=ls,
                      label=geneName+' ROC curve (area = %0.2f)' % roc_auc)
        else:
            plt.plot(fpr, tpr, color=color,linewidth=3,ls=ls,
                      label=geneName+' '+project+' ROC curve (area = %0.2f)' % roc_auc)
        
        plt.plot([0, 1], [0, 1], color='black', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate',fontsize=16)
        plt.ylabel('True Positive Rate',fontsize=16)
        plt.legend(loc="lower right",fontsize=16)
        plt.axis('square')
        if title is None:
            plt.title(geneName,fontsize=16)
        else:
            plt.title(title,fontsize=16)
    return roc_auc,fpr,tpr
      
#%%

    
def GetSlideLevelResults(cohort,gene,isMisSenseTreatedAsWT=False):
    """
    Load the slide level results that were generated after evaluating all of the
    slide level models.

    Parameters
    ----------
    cohort : string
        Define the cohort. WSI or TCGA
    gene : string
        Define the gene for which to load the results. BAP1, PBRM1 and SETD2
    isMisSenseTreatedAsWT : bool, optional
        If loading the TCGA results, define if you would like to use the missense
        treated as loss or WT. The default is False.

    Returns
    -------
    predScore : np.array
        Prediction scores for each sample.
    isWt : np.array
        Ground truth labels for each sample.
    isFocal : np.array
        For each sample, define whether it is a localized loss sample.
    sampleList : list
        List that contains all of the samples in the same order as prediction 
        scores and labels.
    scoresByFile : list
        Returns the scores tracked directly to their svs files.

    """

    if cohort=='TCGA':
        
        saveDir=os.path.join(projectPaths['DataDir'],projectPaths['ResponseData'],
                             'Slide_Level/TCGA/')
        saveFile = lambda gene,foldNum: os.path.join(saveDir,gene+'_TCGA_Fold'+\
                                                         str(foldNum)+'.pkl')
        
        
        if isMisSenseTreatedAsWT:
            geneticsFile=os.path.join(ROOT_DIR,projectPaths['Slide_Level']['Missense'])
        else:
            geneticsFile=os.path.join(ROOT_DIR,projectPaths['Slide_Level']['Genetics'])
        tcgaGenetics=pd.read_csv(geneticsFile) 
       
        # Standardize genetics format
        genetics=pd.DataFrame({'ID':tcgaGenetics['ID'].values,
                               'is'+gene+'WT':tcgaGenetics['is'+gene+'WT_True'].values,
                               'is'+gene+'Focal':np.zeros(tcgaGenetics.shape[0],dtype=bool)}).set_index('ID')    
        
        # Find svs files corresponding to different patient samples
        # In our cohort only 5 patients had more than one svs for the patient
        tcgaGenetics=tcgaGenetics.set_index('ID')
        sampleToFiles={}
        for patientId in tcgaGenetics.index:
            sampleToFiles[patientId]=[os.path.split(f)[-1] for f in tcgaGenetics.loc[[patientId]]['svs'].values]
            
    elif cohort == 'WSI':
       
        saveDir=os.path.join(projectPaths['DataDir'],projectPaths['ResponseData'],
                             'Slide_Level/WSI/')
        
        saveFile = lambda gene,foldNum: os.path.join(saveDir,gene+'_'+'WSI_Fold'+str(foldNum)+'.pkl')    
        
        
        foldsIdx=os.path.join(ROOT_DIR,projectPaths['Data']['FoldsIdx'])
        with open(foldsIdx, 'rb') as f:
            folds = pickle.load(f)
        validIdx=folds[3]
        allSampleFile=os.path.join(ROOT_DIR,projectPaths['Data']['AllSamples'])
        allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)
        holdoutSamples=allSamples.iloc[validIdx]
        
        # Standardize genetics format
        genetics=pd.DataFrame({'ID':holdoutSamples['svs'].values,
                               'is'+gene+'WT':holdoutSamples[gene+'_Positive'].values,
                               'is'+gene+'Focal':holdoutSamples[gene+'_Focal'].values}).set_index('ID')
        
        # For WSI each slide corresponds to a different patient, so we use the
        # svs as the sample Id
        sampleToFiles={sampleId:[sampleId] for sampleId in genetics.index}
        
    numberOfFolds=3
    activations={}
    files={}
    for foldNum in range(numberOfFolds):
        [_,_,activations[foldNum],files[foldNum],_]=\
            pickle.load(open(saveFile(gene,foldNum), "rb" ) )
    
    # Confirm that folds results are in same order, and then just use fold 0 order
    assert files[0]==files[1] and files[2]==files[1]
    files=files[0]
    
    # For each svs file combine the activations of bags from all three folds 
    scoresByFile={}
    combinedActivationsByFile={}
    for fileNum,fileName in enumerate(files):
        # This contains the activations for all three folds
        # Activations for each fold is a vector of length = number_of_bags
        acts=[activations[fold][fileNum] for fold in range(numberOfFolds)]
        fileShort=os.path.split(fileName)[-1].replace('.hdf5','.svs')
        # Combine activations from all folds 
        combinedActivations=np.concatenate(acts)
        combinedActivationsByFile[fileShort]=acts
        # Final activation for each file is  the mean activation across all bags+folds
        scoresByFile[fileShort]=np.nanmean(combinedActivations)
     
    
    scoresBySample={}
    for sample in sampleToFiles:
        scores=[]
        for file in  sampleToFiles[sample]:
            if file in scoresByFile:
                scores.append(scoresByFile[file])
            else:
                print(file+ ' missing')
        if len(scores)==0:
            scoresBySample[sample]=np.NAN
            print(file+ ' empty')
        else:
            scoresBySample[sample]=np.nanmin(scores)
    
    sampleList=[sample for sample in scoresBySample]
    isWt=[]
    isFocal=[]
    predScore=[]
    for sample in sampleList:
        
        assert len(np.unique(genetics.loc[[sample]]['is'+gene+'WT'].values)) ==1
        isWt.append(genetics.loc[[sample]]['is'+gene+'WT'].iloc[0]) 
        assert len(np.unique(genetics.loc[[sample]]['is'+gene+'Focal'].values)) ==1
        isFocal.append(genetics.loc[[sample]]['is'+gene+'Focal'].iloc[0])
        predScore.append(scoresBySample[sample])
    
    isWt=np.array(isWt)
    isFocal=np.array(isFocal)
    predScore=np.array(predScore)    

    return predScore,isWt,isFocal,sampleList,scoresByFile



def Get_TMA_Response(cohort,isNormalized=True,
                     showSlideImages=False,showPunchImages=False):
    """
    Load the region scores for the TMA cohorts. This will load them into a pandas
    directory.

    Parameters
    ----------
    cohort : string
        Specify if WSI/PDX1/TMA1/TMA2.
    isNormalized : bool, optional
        Indicate if colorNormalized. The default is True.
    showSlideImages : bool, optional
        The default is False.
    showPunchImages : bool, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    cohortOutput
    """
    
    
    # Load directories and variables
    layoutDir=os.path.join(ROOT_DIR, projectPaths['Region_Level']['TMALayout'])
    tumorDir=os.path.join(projectPaths['DataDir'],projectPaths['Tumor'],'TMA/')
    tumorStride=trainParams['TumorParams']['tumorStride'] #16 by default
    tumorPatchSize=trainParams['TumorParams']['tumorPatchSize'] # 256 by default 
    tessellationStride=trainParams['Tessellation']['stride']  #100 by default
    cohortDsf=trainParams['DSF']['TMA']  #Loads dsf for TMA cohorts in dict

    assert cohort in cohortDsf
    dsf=cohortDsf[cohort]
    
    
    if isNormalized:
       
        activationsDir=os.path.join(projectPaths['DataDir'],projectPaths['ResponseData'],
                                    'Region_Level','TMA_Norm')
    else:
       
        activationsDir=os.path.join(projectPaths['DataDir'],projectPaths['ResponseData'],
                                    'Region_Level','TMA_NoNorm')
    
            
    layoutFile=os.path.join(layoutDir, cohort+'_Layout.csv')
    fullLayout=pd.read_csv(layoutFile).set_index('SVS')
    uniqueSvs=np.unique(fullLayout.index)
    outDfList=[]
    
    #Loop over all svs slides
    for svs in uniqueSvs:
        # Load the corresponding layout file for each svs
        svsLayout=fullLayout.loc[svs]
        
        # Load gene activations
        activationsFile=os.path.join(activationsDir, svs.replace('.svs','_avg.pkl'))
        _,activations=pickle.load(open(activationsFile,'rb'))
        activations=activations[:,:,1]
        
        # Load Tumor Mask and resize to activations
        tumorPkl=os.path.join(tumorDir,svs.replace('.svs','.pkl'))
        regionMask,_=pickle.load(open(tumorPkl ,'rb'))
        tissueMask=regionMask>0 # Anything that is not background
        padding = int(np.ceil(np.round(tumorPatchSize/tumorStride)))      
        tissueMaskResized=np.uint8(resize(np.pad(np.float32(tissueMask), [[0, padding],[0, padding]]),
                               activations.shape))
        
        # Branch used to visualize activations and tissue masks
        if showSlideImages:
            plt.figure(figsize=(20,20))
            plt.imshow(activations)
            plt.imshow(tissueMaskResized,alpha=0.4,cmap=ListedColormap(['w','r']))
            plt.title(svs)
            
        activationList=[]
        for punchCounter in range(svsLayout.shape[0]):
            # svsLayout cotains punch level information for all slides
            punchInfo=svsLayout.iloc[punchCounter]
            
            
            # Only preserve punches that are not blacklisted, have acceptable
            # BAP1 staining calls and have metadata
            isGood=punchInfo.hasMetadata and \
                punchInfo.isBAP1WT in [0,1] and \
                    (not punchInfo.isBlackListed)
            
            
            if isGood:
                
                # Get Corner of Punch, and rescale position
                cornerX=int(punchInfo.CornerX/(dsf*tessellationStride))
                cornerY=int(punchInfo.CornerY/(dsf*tessellationStride))
                
                height=int(punchInfo.Height/(dsf*tessellationStride))
                width=int(punchInfo.Width/(dsf*tessellationStride))
                
                # Get Activation and Mask Maps for Punch
                punchActivation=activations[cornerY:(cornerY+height),
                                           cornerX:(cornerX+width)]
                punchMask=tissueMaskResized[cornerY:(cornerY+height),
                                           cornerX:(cornerX+width)]
                
                # Visualize each punch activation and tissue mask
                if showPunchImages:
                    plt.figure(figsize=(15,5))
                    plt.subplot(1,3,1)
                    plt.imshow(punchActivation,vmin=0,vmax=1)
                    plt.subplot(1,3,2)
                    plt.imshow(punchMask,cmap=ListedColormap(['w','r']))
                    plt.subplot(1,3,3)
                    plt.imshow(punchActivation,vmin=0,vmax=1)
                    plt.imshow(punchMask,cmap=ListedColormap(['w','r']),alpha=0.8)
                    plt.show()
                
                punchScore=np.mean(punchActivation[punchMask>0])
            else:
                punchScore=np.NAN
            activationList.append(punchScore)
            
        outDf=svsLayout.copy()
        outDf['BAP1_Response']=activationList
        outDfList.append(outDf)
    
    # Concatenate data for each slide to an output file
    cohortOutputRaw=pd.concat(outDfList)    
    
    # Filter out those with invalid (np.NAN) BAP1 response
    cohortOutput=cohortOutputRaw[np.logical_not(np.isnan(cohortOutputRaw['BAP1_Response']))]
    return cohortOutput

def TMA_Patient_Level_Response(responseInfo):
    """ 
    Combine TMA response data to show patient level predictions and labels
    

    Parameters
    ----------
    responseInfo : dataframe
        Response info dataframe that contains punch level predictions

    Returns
    -------
    simpleDf : dataframe
        Dataframe containing patient level information 

    """
    
    
    patientIDs=np.float64(responseInfo.PatientID.values)
    uniqueIDs=np.unique(patientIDs)
    
    patientActs,patientGT,patientGrades=[],[],[]
    for patient in uniqueIDs:
        coreIdxs=np.where(patientIDs==patient)[0]
        patientActs.append(np.nanmin(responseInfo.BAP1_Response.values[coreIdxs]))
        
        patientGT.append(np.all(responseInfo.isBAP1WT.values[coreIdxs]))
        patientGrades.append(np.nanmax(responseInfo.Grade.values[coreIdxs]))
        
    simpleDf=pd.DataFrame({'PatientID':uniqueIDs,
        'isBAP1WT': patientGT,
        'BAP1_Response': patientActs,
        'Grade': patientGrades})
    
    return simpleDf
                         

def Load_Test_Samples(focalStatus,geneToAnalyze):
    """ 
    Load test samples from WSI Testing. Returns svs file names only
    

    Parameters
    ----------
    focalStatus : str
        focal / nonFocal.
    geneToAnalyze : str
        BAP1, PBRM1, SETD2.

    Returns
    -------
    'focal'
        list of svs filenames
    'nonFocal
        list of svs filenames, list of boolean values indicating if BAP1 positive or not

    """
    foldsIdx=os.path.join(ROOT_DIR, projectPaths['Data']['FoldsIdx'])
    with open(foldsIdx, 'rb') as f:
        folds = pickle.load(f)
    validIdx=folds[3]
    
    # Load the testing data into memory
 
    allSampleFile = os.path.join(ROOT_DIR, projectPaths['Data']['AllSamples'])
    allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)
    
    testSamples=allSamples.iloc[validIdx]
    if focalStatus=='nonFocal':
        testNonFocalSamples=testSamples.iloc[np.where(testSamples[geneToAnalyze+'_Focal'].values==False)[0]]
        testSampleToClassNF=np.uint8(testNonFocalSamples[geneToAnalyze+ '_Positive'].values)
        return testNonFocalSamples.svs.values,testSampleToClassNF
    else:
        testFocalSamples=testSamples.iloc[np.where(testSamples.BAP1_Focal.values==True)[0]]
        return testFocalSamples.svs.values 
    
def Load_Train_Samples(fold,focalStatus,geneToAnalyze):
    """ Load training samples from WSI. Returns svs file names only 
    

    Parameters
    ----------
    fold : TYPE
        DESCRIPTION.
    focalStatus : TYPE
        DESCRIPTION.
    geneToAnalyze : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    foldsIdx=os.path.join(ROOT_DIR, projectPaths['Data']['FoldsIdx'])
    with open(foldsIdx, 'rb') as f:
        folds = pickle.load(f)
    
    validIdx=folds[fold]
    
    # Load the training data into memory
 
    allSampleFile = os.path.join(ROOT_DIR, projectPaths['Data']['AllSamples'])
    allSamples = pd.read_csv(allSampleFile).drop(['Unnamed: 0'], axis=1)
    
    trainSamples=allSamples.iloc[validIdx]
    if focalStatus=='nonFocal':
        trainNonFocalSamples=trainSamples.iloc[np.where(trainSamples[geneToAnalyze+'_Focal'].values==False)[0]]
        trainSampleToClassNF=np.uint8(trainNonFocalSamples[geneToAnalyze+ '_Positive'].values)
        return trainNonFocalSamples.svs.values,trainSampleToClassNF
    else:
        trainFocalSamples=trainSamples.iloc[np.where(trainSamples.BAP1_Focal.values==True)[0]]
        return trainFocalSamples.svs.values 

#%%
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy. Defined here to be able to 
    use without having the use the keras implementation
    
    # Inputs
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Outputs
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
        classes : int : number of classes
        patchSize : int : default is 224
    Outputs: 
        finalPatchMask : array : resized patch mask to be same size as original patches
    """
    patchMaskOHE=to_categorical(patchMask,num_classes=classes)
    resizedPatchMask=np.zeros((dimensions[0],dimensions[1],classes))
    for layer in range(classes):
        resizedPatchMask[:,:,layer]=resize(patchMaskOHE[:,:,layer],(dimensions[0],dimensions[1]))
    finalPatchMask=np.argmax(resizedPatchMask,axis=2)
    return finalPatchMask

def Plot_Patch(svs,coordinates):
    """Plots a single patch from given coordinates
    This function was used to generate some of the patches used in the supp figures
    Parameters
    ----------
    svs : TYPE
        DESCRIPTION.
    coordinates : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    svsDir=os.path.join(projectPaths['DataDir'],
                        projectPaths['Data']['ImageData'],'WSI/')
    svsFile=svsDir+svs
    slide=oSlide.open_slide(svsFile)
    
    patch=slide.read_region((coordinates[0],coordinates[1]),0,(224,224))
    
    plt.imshow(patch)
    plt.axis('off')
    
    
    
#%% New class defined to calculate the WSI regional response scores
    
class RegionalResponse:
    
    """
    This class contains all of the necessary functions to load the regional 
    response for the WSI samples.
    """
    def __init__(self, geneToAnalyze,tessellationStride,minTumorThreshold):
        self.geneToAnalyze=geneToAnalyze
        self.minTumorThreshold=minTumorThreshold
        self.tessellationStride=tessellationStride
        self.gridSizeInMicrons=1000
        

    def GetWsiSamples(self, fold='holdout',setType='test'):
        """
        Load the correct WSI samples. Individual folds or holdout set
        
        Parameters
        ----------
        fold : string or int
            Define the fold to load. May also be holdout set
        setType : string
            Define which set to load (train vs test). Only applicable to folds
        
        Returns
        -------
        sampleInfo : dataframe 
            Dataframe containing all sample info for the indicated fold and 
            setType.
        """
        assert fold in [0,1,2,'holdout']
        assert setType in ['train','test']
        
     
        foldsFile=os.path.join(ROOT_DIR, projectPaths['Data']['FoldsIdx'])
        foldIndices=pickle.load(open(foldsFile,'rb'))
        folds={}
        folds[0]={'train':np.concatenate((foldIndices[1],foldIndices[2])),'test':foldIndices[0]}
        folds[1]={'train':np.concatenate((foldIndices[0],foldIndices[2])),'test':foldIndices[1]}
        folds[2]={'train':np.concatenate((foldIndices[0],foldIndices[1])),'test':foldIndices[2]}
        folds['holdout']={'train':[],'test':foldIndices[3]} # never used throughout the repo
    
        sampleIdx=folds[fold][setType]
    
        sampleInfoFile=os.path.join(ROOT_DIR, projectPaths['Data']['AllSamples'])       
        allSamples = pd.read_csv(sampleInfoFile).drop(['Unnamed: 0'], axis=1)
           
        sampleInfo=allSamples.iloc[sampleIdx].reset_index(drop=True)
        
        
        return sampleInfo
    
    def GetTrueAndPredMaps(self, svs,geneToAnalyze,isWt,isFocal,showFigures=False):
        """
        This function loads the activation maps and in the case of localized loss
        cases, the ground truth maps.

        Parameters
        ----------
        svs : string
            svs file.
        geneToAnalyze : string
            BAP1, PBRM1 or SETD2.
        isWt : np,array
            Ground truth label.
        isFocal : np.array
            Localized loss status.
        showFigures : bool, optional
            Display the activation maps, tumor masks, and ground truth maps for 
            corresponding files as they are loaded. The default is False.

        Returns
        -------
        predActivationMap : np.array
            Prediction activation map for svs.
        trueLossMap : np.array
            For focal samples, display the true loss map.
        mpp : TYPE
            Micros per pixel. Metadata from the svs file

        """
        
        wsiActivationsBaseDir=os.path.join(projectPaths['DataDir'], 
                                           projectPaths['ResponseData'],
                                           'Region_Level/WSI/')
        
        wsiSvsDir=os.path.join(projectPaths['DataDir'], 
                               projectPaths['Data']['ImageData'],
                               'WSI/')
        

        wsiTumorMasksDir=os.path.join(projectPaths['DataDir'],
                                      projectPaths['Tumor'],'WSI/')
        wsiFocalMasksDir=os.path.join(ROOT_DIR,
                                      projectPaths['Region_Level']['XmlDir'])
        
        activationFile=os.path.join(wsiActivationsBaseDir,geneToAnalyze,
                                    svs.replace('.svs','_avg.pkl'))
        _,predActivationMap=pickle.load(open(activationFile,'rb'))
        predActivationMap=predActivationMap[:,:,1]
        
        
        tumorFile=os.path.join(wsiTumorMasksDir,svs.replace('.svs','.pkl'))
        tumorMask,tumorLabels=pickle.load(open(tumorFile,'rb'))
        tumorMask=tumorMask==tumorLabels.index('tumor')
            
        tumorMask=cv2.resize(np.uint8(tumorMask),
                             (predActivationMap.shape[1],predActivationMap.shape[0]))
        slide=oSlide.open_slide(os.path.join(wsiSvsDir,svs))
        mpp=np.float32(slide.properties['openslide.mpp-x']) # direct slide properties mpp
         #microns to millimeter
        
        if isFocal:
            assert geneToAnalyze =='BAP1' ,"PBRM1 and SETD2 Focal Analysis not supported"
        
        
            
            lossAnnoFile=os.path.join(wsiFocalMasksDir,svs.replace('.svs','.xml'))
            lossMask, maskDict= pg.MaskFromXML(lossAnnoFile,'Focal_BAP1',slide.dimensions,
                                               downSampleFactor=self.tessellationStride,
                                               distinguishAnnosInClass=False)
            if lossMask.shape!=predActivationMap.shape: 
                lossMask=cv2.resize(np.uint8(lossMask),
                                    (predActivationMap.shape[1],predActivationMap.shape[0]))
        
            trueLossMap=np.uint8(tumorMask)*2
            trueLossMap[np.logical_and(lossMask!=0,trueLossMap)]=1
            
        else:
            
            trueLossMap=tumorMask.copy()
            trueLossMap[tumorMask>0]=isWt+1
        
        if showFigures:
            plt.figure(figsize=(20,20))
            plt.imshow(predActivationMap)
            plt.imshow(trueLossMap,vmin=0,vmax=2,cmap=ListedColormap(['w','r','g']),alpha=0.5)
        
        return predActivationMap,trueLossMap,mpp


    def Get_WSI_Response(self,restrictToFocal=False,showFigures=False):
        """
        This function returns the scores for each region of all WSI samples

        Parameters
        ----------
        restrictToFocal : bool, optional
            Load only focal samples. The default is False.
        showFigures : bool, optional
            Plot the gridded images as they are being analyzed. 
            The default is False.

        Returns
        -------
        trueClasses : np.array
            Ground truth labels for all of the regions analyzed.
        predActivations : np.array
            Activations scores for all regions in all WSI samples. Each score 
            represents a single region in a sample.
        svsFiles : list
            List of svs files analyzed.

        """
        holdoutInfo=self.GetWsiSamples(fold='holdout')           
        
        if self.geneToAnalyze == 'BAP1':
            
            if restrictToFocal:
                isFocal=holdoutInfo[self.geneToAnalyze+'_Focal']
                samplesToAnalyze=holdoutInfo[isFocal].reset_index(drop=True)
            else:
                samplesToAnalyze=holdoutInfo
        else:
            isNonFocal=np.logical_not(holdoutInfo[self.geneToAnalyze+'_Focal'])
            samplesToAnalyze=holdoutInfo[isNonFocal].reset_index(drop=True)
            
        samplesToAnalyze=samplesToAnalyze.set_index('svs')
        svsList=samplesToAnalyze.index.values
        
        trueClasses=[]
        predActivations=[]
        svsFiles=[]
        for svs in tqdm(svsList):
            
            isFocal= samplesToAnalyze.loc[svs][self.geneToAnalyze+'_Focal']
            isWt= int(samplesToAnalyze.loc[svs][self.geneToAnalyze+'_Positive'])
            
            # Get predicted activation and ground-truth maps (resized to have size of activation map.) 
            # In the ground truth map 0 = Non-Tumor, 1 = Loss Tumor, 2 = WT Tumor
            
            if restrictToFocal and showFigures:
                predActivationMap,trueLossMap,micronsPerPixel=\
                    self.GetTrueAndPredMaps(svs,self.geneToAnalyze,isWt,isFocal,
                                    showFigures=showFigures)
            else:
                predActivationMap,trueLossMap,micronsPerPixel=\
                    self.GetTrueAndPredMaps(svs,self.geneToAnalyze,isWt,isFocal,
                                    showFigures=False)
            
            # We divide the tissue in 1mmx1mm grids. Convert this into our units 
            # in out maps by using the microns_per_pixel of the original image
            # and the downsampling we get in calculations of the activation 
            gridSize=int(self.gridSizeInMicrons/(micronsPerPixel*self.tessellationStride))
            
            # Crop the Maps to have integral number of grid points
            gridR,gridC=np.uint32(gridSize*np.floor(np.array(predActivationMap.shape)/gridSize))
            predActivationMapCropped=predActivationMap[:gridR,:gridC]
            trueLossMapCropped=trueLossMap[:gridR,:gridC]
            
            # Average Activation and Composition in Grid
            tissueMaskCropped=trueLossMapCropped!=0
            gridActMap=block_reduce(np.array(predActivationMapCropped*tissueMaskCropped),
                               block_size=(gridSize,gridSize),func=np.mean)
            gridTrueMap=block_reduce(to_categorical(trueLossMapCropped,num_classes=3),
                                    block_size=(gridSize,gridSize,1),func=np.mean)
            
            # IDentify Tumor Grid Cells
            fracTumor=np.sum(gridTrueMap[:,:,1:],axis=-1)
            isTumor=fracTumor>self.minTumorThreshold
            
            gridActMap=gridActMap/fracTumor
            
            # Determine WT fraction in a grid cell as ratio of WT to Tumor Pixels
            fracWT=gridTrueMap[:,:,2]
            wtScore=fracWT/fracTumor
            
            # Return Results on Tumor Grid Cells
            trueClassesSvs= np.round(wtScore[isTumor])
            predActivationsSvs= gridActMap[isTumor]
            sampleNameSvs= [svs]*len(predActivationsSvs)
            
            trueClasses.append(trueClassesSvs)
            predActivations.append(predActivationsSvs)
            svsFiles.append(sampleNameSvs)
            
            if showFigures:
                # Implementation using old code
                # Should be easy to also generate ground truth classes in this way
                x=np.arange(0,predActivationMap.shape[0],gridSize)
                y=np.arange(0,predActivationMap.shape[1],gridSize)
            
                
                regionScores=[]
                classLabels=[]
                classLabelsScores=[]
                tumorMask=trueLossMap>0
                x1=0
                for row in x:
                    y1=0
                    for col in y:
                        gridZone=tumorMask[x1:row+gridSize,y1:col+gridSize]
                        if 0 not in gridZone.shape:
                            if np.mean(gridZone)>self.minTumorThreshold:
                                tumorZone=gridZone
                                predictionZone=predActivationMap[x1:row+gridSize,y1:col+gridSize]
                                predictedAvg=np.mean(predictionZone[np.where(tumorZone==1)])
                                groundTruthZone=trueLossMap[x1:row+gridSize,y1:col+gridSize]
                                classLabelsScores.append((np.mean(groundTruthZone[np.where(groundTruthZone!=0)])))
                                classLabels.append(np.round(np.mean(groundTruthZone[np.where(groundTruthZone!=0)])))
                                regionScores.append(predictedAvg)       
                        y1=y1+gridSize
                    x1=x1+gridSize
    
                plt.figure(figsize=(10,10))
                plt.subplot(1,2,1)
                plt.scatter(regionScores,predActivationsSvs)
                plt.xlabel('Old For-Loop Calculation')
                plt.ylabel('New BlockReduce Calculation')
                plt.title('Activations')
                plt.plot([0,1],[0,1],'--k')
                plt.axis('square')
                plt.subplot(1,2,2)
                plt.scatter(np.array(classLabelsScores)-1,wtScore[isTumor])
                plt.plot([0,1],[0,1],'--k')
                plt.xlabel('Old For-Loop Calculation')
                plt.ylabel('New BlockReduce Calculation')
                plt.title('Ground Truth')
                plt.axis('square')
            
        trueClasses=np.concatenate(trueClasses)    
        predActivations=np.concatenate(predActivations)
        svsFiles=np.concatenate(svsFiles)
        return np.float64(trueClasses), np.float64(predActivations), svsFiles
