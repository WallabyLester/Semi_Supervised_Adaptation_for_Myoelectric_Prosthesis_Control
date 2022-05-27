# EMG Classification for Myoelectric Control
## Collaboration with Northwestern MSR, Shirley Ryan AbilityLab, and Coapt, LLC

`Data_Analysis` - contains various Jupyter Notebooks which present data trends and information for creating rules in classifying EMG data and relabeling said data. The accompanying `.py` files are the text files for the Jupyter Notebooks. If you would like to use this same format, you will need to install Jupytext to enable editing of notebooks in text files and accurate version control. 

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
