## Gene_Model_Training

This folder contains the code to train the models to predict BAP1/SETD2/PBRM1 status from H&E images at both the slide and regional levels. Please note that all models are trained using a three fold stratified cross validation technique and that each gene is trained independently. Consequently, for each gene,  three different models must be trained (one for each fold). The models will save in the corresponding subfolders of the `/Models/` folder.

Note: All command below assume you are executing from within the singularity shell and are located in the root directory of this repository. See [here](../Environment_README.md) for instructions. Additionally the instructions here assume you have already created appropriate paths, downloaded all the raw image data, and extracted image patches as indicated [here](../Data_Instructions.md) and [here](../Data_Preparation/README.md).

### Slide Level Training 
To train the slide level models for BAP1 fold 0 run the following command:

	python Gene_Model_Training/Training_Slide_Model.py --geneToAnalyze BAP1 --foldNum 0

Change the --geneToAnalyze and --foldNum arguments to all combinations of genes (BAP1/PBRM1/SETD2) and folds (0/1/2) respectively to train all models.

### Region Level Training
To train the region level models for BAP1 fold 0 run the following command:

	python Gene_Model_Training/Training_Region_Model.py --geneToAnalyze BAP1 --foldNum 0

Similar to the slide level models, change the --geneToAnalyze and --foldNum arguments to all combinations of genes and folds respectively to train all region level models.

