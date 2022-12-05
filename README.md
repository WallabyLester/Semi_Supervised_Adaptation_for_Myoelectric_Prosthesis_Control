# Semi_Supervised_Adaptation_for_Myoelectric_Prosthesis_Control
## Collaboration with Northwestern MSR, Shirley Ryan AbilityLab, and Coapt, LLC

## Overview
The goal of this project was to utilize the existing EMG sensor device designed by Coapt in combination with virtual reality (VR) games to improve EMG-based classification. Virtual game data was used from prosthesis users playing two of Coapt's VR games. Data will not be available with this repository. Classification was performed with an adaptive LDA model used by Coapt and recreated for this project. (Refer to: Vidovic et al. 2015 Improving the Robustness of Myoelectric Pattern Recognition for Upper Limb Prostheses by Covariate Shift Adaptation) This repository contains various files used for data analysis with EMG data and the adaptive LDA and rules used. 

## Contents
`Classification` - contains Jupyter Notebooks for applying an adaptive LDA on virtual game data along with comparisons to an unadapted LDA and relabeled data from the applied rules. The accompanying `.py` files are the text files for the Jupyter Notebooks. Also contains the adaptive LDA.

`Cloud` - contains files for applying the adaptive LDA and rules in a cloud database. 

`Data_Analysis` - contains Jupyter Notebooks which present data trends and information for creating rules in classifying EMG data. The accompanying `.py` files are the text files for the Jupyter Notebooks. 

`Rules` - contains Jupyter Notebooks using the data analysis to find the rules and relabel virtual game data based off of them. The accompanying `.py` files are the text files for the Jupyter Notebooks. Also contains the rules used in classification. 

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
