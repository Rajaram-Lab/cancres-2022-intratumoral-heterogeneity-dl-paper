#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file is used to generate all of the plots in figure 5 and combine them
together into a single figure. This will call other figure generating files.
The process is wrapped into functions so that it can be called through 
the FigureMasterScript.


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

import os as os 
wrkDir=os.getcwd()
assert os.path.exists(os.path.join(wrkDir,'Parameters/Project_Paths.yaml')),\
    "Current working directory does not match project directory, please change to"+\
        "correct working directory"
ROOT_DIR=wrkDir

import sys 

import yaml


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

with open(os.path.join(ROOT_DIR,'Parameters/Figure_Params.yaml')) as file:
    figParams = yaml.full_load(file)
sys.path.insert(0, ROOT_DIR)

import Figure_Generation.Figure5ab_Plots as fig5ab
import Figure_Generation.Grade_Survival as gs

# Define plotting parameters
fontSize=figParams['fontsize']
mutColors=figParams['colors']
wtColors=figParams['wtColors']
geneList=figParams['geneList']



#%% Figure 5
def PlotFigure5():
    fig = plt.figure(figsize=(26,15))
    # set up subplot grid
    gridspec.GridSpec(8,6)
    
    # small subplot 1 - Gene kernel density
    plt.subplot2grid((8,6), (0,0), colspan=2, rowspan=3)
    fig5ab.ClassificationIndependence(['BAP1','PBRM1'],figPlot=True,plotNum=0)
    
    # large subplot 2 - Gene Scatterplot
    plt.subplot2grid((8,6), (3,0), colspan=2,rowspan=5)
    fig5ab.ClassificationIndependence(['BAP1','PBRM1'],figPlot=True,plotNum=1)
    
    # small subplot 4 - BAP1 nuclear features density 1
    plt.subplot2grid((8,6), (0,2), rowspan=4, colspan=1)
    fig5ab.NuclearFeatureDensity(plotNum=0)
    
    # small subplot 5 - PBRM1 nuclear features density 1
    plt.subplot2grid((8,6), (4,2), rowspan=4, colspan=1)
    fig5ab.NuclearFeatureDensity(plotNum=1)
    
    # small subplot 6 - Tumor Grade
    plt.subplot2grid((8,6), (0,3), colspan=2,rowspan=4)
    gs.Plot_Grades('TMA2_Patient',groupGrades=True,useStatAnnotPvals=True)
    
    #  subplot 7 - Survival
    plt.subplot2grid((8,6), (4,3), colspan=2,rowspan=5)

    gs.Plot_Survival('TMA2',survivalType='disease-specific')
    plt.title('Disease-Specific Survival',fontsize=16)
    fig.tight_layout()
# PlotFigure5()
