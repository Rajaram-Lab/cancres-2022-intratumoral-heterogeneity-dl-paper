This folder contains the preprocessing applied to the raw image files to allow us to train our models. 

Note: All command below assume you are executing from within the singularity shell and are located in the root directory of this repository. See [here](../Environment_README.md) for instructions on how to achieve this. Additionally the instructions here assume you have already downloaded all the raw image data, and will save output results in sub-folders of data folder. So please ensure you have created the correct directories and updated file paths as indicated [here](../Data_Instructions.md). 


## Tumor Mask Generation

The vast majority of our analysis is applied only to pixels considered to be in the tumor area (as opposed to normal, blood, non-tissue etc) within a slide. Thus the first preprocessing step is to apply a previously model to the slides to generate masks identifying the tumor regions therein.

From the root working directory, run the following command:

	python Data_Preparation/Tumor_Mask_Generation.py

This command will generate tumor masks for all of the cohorts found in `Raw_Slide_Images/` and it will save them in the corresponding section of the  `Intermediate_Results/Tumor/` folder. Because the three TMA cohorts [TMA1, TMA2, PDX1] do not contain many SVS files, the SVS files and tumor masks are grouped into folders simply named `TMA` within their respective directory (i.e.`Raw_Slide_Images/TMA/` and `Intermediate_Results/Tumor/TMA/`). Note that running the tumor models will require a GPU. Additionally, note that the command by default will loop through all of the cohorts for simplicity's sake, but that will cause extended compute times for the WSI and TCGA cohorts. To avoid such issues, you  may want to partition the task into different jobs like the examples shown below (please read code for further explanation). `sampleEnd` and `sampleStart` parameters should be within the bounds of the number of files for each cohort as shown in the table above. 

	python Data_Preparation/Tumor_Mask_Generation.py --singleCohort WSI --sampleStart 0 --sampleEnd 431
	python Data_Preparation/Tumor_Mask_Generation.py --singleCohort WSI --sampleStart 431 --sampleEnd 862
	python Data_Preparation/Tumor_Mask_Generation.py --singleCohort WSI --sampleStart 862 --sampleEnd 1292

## Patch Generation
Our analysis is largely applied to image patches extracted from the tumor regions. To generate these patches run the following three commands:	

	python Data_Preparation/WSI_PatchGeneration.py
	python Data_Preparation/LocalizedLoss_PatchGeneration.py
	python Data_Preparation/TCGA_PatchGeneration.py

These commands will generate patches for WSI, WSI Localized Loss, and TCGA samples respectively. WSI localized loss cases require manual annotations for ground truth labels, so they are generated in their own script. All these commands will save all generated patches in the corresponding `Patch_Data/` folder.  Unlike tumor mask generation, patch generation does not require any GPU resources and thus will have significantly faster compute times for each file. Although not necessary, it is also possible to partition each job further using methods similar to those mentioned above. The examples below apply to the WSI cohort, but can be changed to optimize TCGA patch generation as well. Please note that the Localized_Loss_PatchGeneration.py script will only process localized loss samples (n=20), so it does not support sampleStart and sampleEnd arguments.

	python Data_Preparation/WSI_PatchGeneration.py --sampleStart 0 --sampleEnd 431
	python Data_Preparation/WSI_PatchGeneration.py --sampleStart 431 --sampleEnd 862
	python Data_Preparation/WSI_PatchGeneration.py --sampleStart 862 --sampleEnd 1292

