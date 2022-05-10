## Evaluation
The following will explain how to evaluate the region and slide level models on different test datasets. As there are 3 folds for each gene/model type, the evaluation is a two step process. First, we make predictions based on each of the three folds for a gene, and then the final evaluation result is the average of the individual fold predictions. 

Note: All command below assume you are executing from within the singularity shell and are located in the root directory of this repository. See [here](../Environment_README.md) for instructions. Additionally the instructions here assume you have already created appropriate paths, downloaded all the raw image data, and extracted image patches as indicated [here](../Data_Instructions.md) and [here](../Data_Preparation/README.md).

### Slide Level Evaluation

Slide level predictions are perform on the WSI (held out portion) and TCGA cohorts. The following command will evaluate all three slide level fold models for BAP1 in the WSI cohort: 

	python Gene_Model_Evaluation/Evaluate_Slide_Level.py --cohort WSI --gene BAP1

Replace `--gene` parameter by SETD2/PBRM1 and the `--cohort` parameter by TCGA as required. Note, that the results for the three folds will be saved to the `/Intermediate_Results/Slide_Level`; averaging of these fold results is performed in downstream analysis. 

### Region Level Evaluation

Regions level predictions are perform on the WSI (held out portion) and TMA cohorts (TMA1,TMA2 and PDX1). The following command will evaluate all three slide level fold models for BAP1 in the WSI cohort: 

	python Gene_Model_Evaluation/Region_Level_Inference.py --cohort WSI --gene BAP1

Prediction results will be saved to the `/Intermediate_Results/Region_Level`folder. Replace `--gene` parameter by SETD2/PBRM1 for the WSI cohort and the `--cohort` parameter by TMA1/TMA2/PDX1 as required (note only BAP1 was evaluated on the TMA cohorts for the paper). This script has an additional parameter called `performNorm` that is set to **0** by default. This parameter indicates whether the script will perform normalization on the samples before the evaluation (0=No, 1=Yes). Although our supplementary figures have analysis of the TMA datasets without normalization, the main results use the TMA cohorts evaluation with normalization. To generate the main figure results for TMA cohorts, you must include the additional argument `--performNorm 1` in the command line.

After all of the models & cohorts have been evaluated, run the following command to average all of the individual intermediate results:

	python Gene_Model_Evaluation/Average_Region_Inference.py
