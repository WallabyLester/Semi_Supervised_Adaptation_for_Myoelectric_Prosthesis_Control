# Semi-Supervised Adaptation for Myoelectric Prosthesis Control
## Collaboration with Northwestern MSR, Shirley Ryan AbilityLab, and Coapt, LLC

## Overview
The goal of this project was to utilize the existing EMG sensor device designed by Coapt in combination with virtual reality (VR) games to improve EMG-based classification. Virtual game data was used from prosthesis users playing two of Coapt's VR games. Data will not be available with this repository. Classification was performed with an adaptive LDA model used by Coapt and recreated for this project. (Refer to: Vidovic et al. 2015 Improving the Robustness of Myoelectric Pattern Recognition for Upper Limb Prostheses by Covariate Shift Adaptation) This repository contains various files used for data analysis with EMG data and the adaptive LDA and rules used. 

## Contents
`Classification` - contains Jupyter Notebooks for applying an adaptive LDA on virtual game data along with comparisons to an unadapted LDA and relabeled data from the applied rules. The accompanying `.py` files are the text files for the Jupyter Notebooks. Also contains the adaptive LDA.

- Adaptive LDA All Subjects: Uses an unadapted LDA, adapted LDA, and adapted LDA with rules on all the subjects. 
- Adaptive LDA Aug Rule: Performs comparisons using an unadapted LDA, adapted LDA, and adapted LDA with the augmented rule per subject for their given number of games.
- Adaptive LDA Base Rule: Performs comparisons using an unadapted LDA, adapted LDA, and adapted LDA with the base rule per subject for their given number of games.
- Adaptive LDA Tested Relabeled: Performs comparisons using an unadapted LDA, adapted LDA, and adapted LDA with the base rule per subject for their given number of games. Tests all three classification models on relabeled data. 
- LDA: Unadapted LDA class with fit and prediction functions.
- LDA Adaptive: Adapted LDA class with fit, prediction, and updating mean and covariance functions. Includes a function for loading data in from virtual game files.

`Cloud` - contains files for applying the adaptive LDA and rules in a cloud database. 

- Adaptive Cloud Analysis: Performs the same adaptive LDA analysis using cloud based data. Enables sending the updated adaptive model back to the cloud.
- Cloud LDA Adaptive: Adapted LDA class with fit, prediction, and updating mean and covariance functions. Includes a function for loading data in from virtual game files. Specific to cloud data format.
- Cloud Rules: Game rules class with base and augmented rules. Returns relabeled data sets. Specific to cloud data format. 

`Data_Analysis` - contains Jupyter Notebooks which present data trends and information for creating rules in classifying EMG data. The accompanying `.py` files are the text files for the Jupyter Notebooks. 

- All Participants: Looks at virtual game data for In the Zone and Simon Says virtual games for all participants. Involves the preramp speed, postramp speed, and consecutive classified motions.
- All Participants V2: Updated virtual game analysis with normalized results instead of raw.
- Consecutive Motions: Deep dive into the consecutive motions in the virtual game data in addition to speeds.
- One Participant: Initial analysis of preramp speed, postramp speed. motion targets, motion predictions, and raw EMG data for just one participant. 

`Rules` - contains Jupyter Notebooks using the data analysis to find the rules and relabel virtual game data based off of them. The accompanying `.py` files are the text files for the Jupyter Notebooks. Also contains the rules used in classification. 

- Per Subject Rule Changes: Uses the base rule and augmented rule on all subjects to visualize the number of data points being relabeled by the rules.
- Rule Finding: Uses consecutive motions with speed and MAV RMS of the data to find the rules. 
- Rule Testing: Uses the base rule and augmented rule on an individual subjects data to test and visualize the effects. 
- Rules: Game rules class with base and augmented rule functions. Returns relabeled data sets.

## Jupytext Instructions
Jupytext was used with the contained Jupyter Notebook to allow for fully synced text files. If you would like to use this same format, you will need to install Jupytext to enable editing of notebooks in text files and clearer version control. 

To install and use Jupytext:
```
pip install jupytext --upgrade

# append this line to your .jupyter/jupyter_notebook_config.py file
# if you do not have this file, create it in your .jupyter directory
c.NotebookApp.contents_manager_class="jupytext.TextFileContentsManager"

# restart the notebook server
jupyter notebook

# to use jupytext with a notebook
# associate a script with your Jupyter Notebook by adding this line to your notebook metadata
"jupytext_formats": "ipynb,py",     # .py can be replaced with most other extensions
```
