#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate_TCGA_Labels is used to create a csv file containing all labels for the
TCGA data. Users can generate standard labels or missense filtered labels.

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

import os , sys
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir
sys.path.insert(0, ROOT_DIR)

import yaml

with open(os.path.join(ROOT_DIR,'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)

import numpy as np

import pandas as pd
import argparse
import yaml

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
    parser = argparse.ArgumentParser(description='Automated job submission for \
        generating standard or missense filtered data on TCGA cohort.')
    
    parser.add_argument('--filterMissense', dest='filterMissense',
                        help='0:False or 1:True',
                        default=0, type=int)
    args = parser.parse_args()
    return args

args=parse_args()
dataDir=projectPaths['DataDir']

# Set up Files
baseTCGADir=os.path.join(dataDir,projectPaths['Data']['ImageData'],'TCGA/')
allSvsFiles=[]
for subdir, dirs, files in os.walk(baseTCGADir,followlinks=True):
    for filename in files:
        filepath = subdir + os.sep + filename
        if filepath.endswith(".svs"):
            allSvsFiles.append(filepath)  

#%%
source='PanCancer'
save=True 
filterMissense=args.filterMissense

downloadedFiles={'PanCancer':os.path.join(ROOT_DIR,
                 projectPaths['Slide_Level']['PanCancer'])}
geneticsFile=downloadedFiles[source]
rawCSV=pd.read_csv(geneticsFile)
sampleIds=rawCSV.columns.values[2:]
geneIdx={'BAP1':3,'PBRM1':4,'SETD2':5}
geneInfo={'BAP1':[],'PBRM1':[],'SETD2':[]}
svsInfo={'BAP1':[],'PBRM1':[],'SETD2':[]}
svsIds={'BAP1':[],'PBRM1':[],'SETD2':[]}
truncSvsName=[]
svsMissingCount=0
missingSamples={'PanCancer':[]}
for svsName in allSvsFiles:
    truncSvsName.append(svsName.rsplit('/',1)[1].rsplit('-',7)[0])

for gene in geneIdx:
    for sample in sampleIds:
        if sample in truncSvsName:
            loc=np.where(np.array(truncSvsName)==sample)[0]
            if loc.size !=1:
                print('Warning: more than one sample ID found')
                # print('Sample ommited from analysis')
                print(gene, loc, sample)
                for i in loc:
                    svsInfo[gene].append(allSvsFiles[i])
                    svsIds[gene].append(sample)
                    if type(rawCSV[sample][geneIdx[gene]])==str:
                        if not filterMissense:
                            geneInfo[gene].append(False)
                        else:
                            if 'Missense' in rawCSV[sample][geneIdx[gene]]:
                                geneInfo[gene].append(True)  #used to consider missesnse cases as WT
                                # geneInfo[gene].append(np.nan)   #uses to remove missense files
                            else:
                                geneInfo[gene].append(False)
                    elif type(rawCSV[sample][geneIdx[gene]])!=str:
                        geneInfo[gene].append(True)
                    else: print(' Not registering label', type(rawCSV[sample][geneIdx[gene]]))
            else:    
                svsInfo[gene].append(allSvsFiles[loc[0]])
                svsIds[gene].append(sample)
                if type(rawCSV[sample][geneIdx[gene]])==str:
                    
                    geneInfo[gene].append(False)
                    
                elif type(rawCSV[sample][geneIdx[gene]])!=str:
                    geneInfo[gene].append(True)
                else: print(' Not registering label', type(rawCSV[sample][geneIdx[gene]]))
        else: 
            print('Sample not found in svs files:',sample)
            svsMissingCount=svsMissingCount+1
            if sample not in missingSamples[source]:
                missingSamples[source].append(sample)
            
    print(svsMissingCount)       
finalSvsSlides=np.array([f.rsplit('/')[-1] for f in svsInfo['BAP1']])

finalSvsIDs=svsIds['BAP1']
geneData=pd.DataFrame({'ID':finalSvsIDs,
          'svs':finalSvsSlides,
          'isBAP1WT_True':geneInfo['BAP1'],
          'isPBRM1WT_True':geneInfo['PBRM1'],
          'isSETD2WT_True':geneInfo['SETD2']})
if save:
    if filterMissense:
        geneData.to_csv(os.path.join(ROOT_DIR,projectPaths['Slide_Level']['Missense']))
    else:
        geneData.to_csv(os.path.join(ROOT_DIR,projectPaths['Slide_Level']['Genetics']))


