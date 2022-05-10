"""
Create_Data_Directory is used to set up the data directory that is needed 
in this project. This file will create the folders and subfolders in the designated
data directory (DataDir) from the Project_Paths.yaml file. Before running this file
please make sure to create an empty folder in directory of your choosing and then 
designate the DataDir parameter to that directory.

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
    
    Paul Acosta, 2022
"""


import os
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir
import yaml

#%% Define data directory
with open(os.path.join(ROOT_DIR, 'Parameters/Project_Paths.yaml')) as file:
    projectPaths = yaml.full_load(file)
dataDir=projectPaths['DataDir']

#%% Create directory structure 

dirList=["Intermediate_Results","Models", "Patch_Data","Raw_Slide_Images"]

for dirName in dirList:
    dirPath=os.path.join(dataDir,dirName)
    os.mkdir(dirPath)
    
    if dirName == 'Intermediate_Results':
        folderList=["Nuclear","Region_Level","Slide_Level","Tumor"]
        
        for folder in folderList:
            folderPath=os.path.join(dirPath,folder)
            os.mkdir(folderPath)
            
            if folder == "Nuclear":
                subfolderList=['Features','Masks']
                for subfolder in subfolderList:
                    subfolderPath=os.path.join(folderPath,subfolder)
                    os.mkdir(subfolderPath)
                    
            elif folder == "Region_Level":
                subfolderList=['TMA_NoNorm','TMA_Norm','WSI']
                for subfolder in subfolderList:
                    subfolderPath=os.path.join(folderPath,subfolder)
                    os.mkdir(subfolderPath)
            
            elif folder == "Slide_Level":
                subfolderList=['TCGA','WSI']
                for subfolder in subfolderList:
                    subfolderPath=os.path.join(folderPath,subfolder)
                    os.mkdir(subfolderPath)
                    
            elif folder == "Tumor":
                subfolderList=['TCGA','TMA','WSI']
                for subfolder in subfolderList:
                    subfolderPath=os.path.join(folderPath,subfolder)
                    os.mkdir(subfolderPath)
            
    elif dirName == 'Patch_Data':
        folderList=["TCGA","WSI","WSI-LL"]
        
        for folder in folderList:
            folderPath=os.path.join(dirPath,folder)
            os.mkdir(folderPath)
            
    elif dirName == 'Raw_Slide_Images':
        folderList=["TCGA","WSI","TMA"]
        
        for folder in folderList:
            folderPath=os.path.join(dirPath,folder)
            os.mkdir(folderPath)
