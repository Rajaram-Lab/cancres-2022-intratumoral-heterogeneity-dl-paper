## Handcrafted Feature Analysis

This folder contains extract hand crafted nuclear features for slides in the WSI cohort, which are later used to train random forest classifiers for predicting BAP1/SETD2/PBRM1 status. This is achieved in the steps indicated below.

Note: All command below assume you are executing from within the singularity shell and are located in the root directory of this repository. See [here](../Environment_README.md) for instructions. Additionally the instructions here assume you have already created appropriate paths, downloaded all the raw image data, and generated tumor masks for the WSI cohort as indicated [here](../Data_Instructions.md) and [here](../Data_Preparation/README.md).

### 1. Extract Nuclear Masks
A previously trained U-Net model trained to classify pixels into (Nucleus/Edge/Background) can be applied to the entire WSI cohort through the following command 

	python Handcrafted/Generate_Nuclear_Masks.py 

All generated masks will be saved to the `Intermediate_Results/Nuclear/Masks/` folder. Note the code offers options to split this analysis into multiple jobs for parallelization.

### 2. Calculate Nuclear Features
For each nucleus in every slide from the WSI cohort, we extract nuclear features (quantifying location, size, shape, texture and color) using the following command:

	python Handcrafted/Extract_Nuclear_Features.py

All generated masks will be saved to the `Intermediate_Results/Nuclear/Features/` folder. Note the code offers options to split this analysis into multiple jobs for parallelization.

### 3. Average Nuclear Features

Next we identify which nuclei are in the tumor regions of a slide, and for each slide represent it by the average (across all tumor-region nuclei) for each feature:

	python Handcrafted/Combine_Features.py

The results across all WSI cohort slides will be saved in a single file  `Intermediate_Results/Nuclear/Features/Combined_Features_Clean.pkl` within the data directory that you were instructed to create [here](Data_Instructions.md).





