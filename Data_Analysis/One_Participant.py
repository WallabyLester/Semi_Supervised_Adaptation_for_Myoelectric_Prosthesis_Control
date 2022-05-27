import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

""" Reads all data in depending on the data location and which ID#

Makes individual Pandas dataframes and places them in a 
python list: `data_list`
"""
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "./Data/ID#31127011_2WProsthUse/VirtualGameData/VirtualArmGames_*"))
data_list = []

for i in csv_files:
    df = pd.read_csv(i)
    data_list.append(df)

def prerampspeed_processing(data):
    """ Does the PreRamp Speed processing to separate into bins based on 
    the data size. 

    Args: 
        data - individual dataframes

    Returns:
        x_array - the bin numbers
        counts - the counts for each speed

    """
    pre_ramp_speed = data['PreRampSpeed']
    
    # Get counts for each speed
    pre_counts = pre_ramp_speed.value_counts()
    pre_index = pre_counts.index.to_numpy().reshape(-1,1)
    pre_vals = pre_counts.to_numpy().reshape(-1,1)
    
    pre_array = np.hstack((pre_index, pre_vals))
    pre_array = pre_array[pre_array[:,0].argsort()]

    # Binning into 5%
    bin_amount = int(np.ceil(len(pre_array) * .05))
    max_range = int(np.floor(len(pre_array) / bin_amount) + 1)
    bins = []
    i = 0
    j = bin_amount
    
    for z in range(0, max_range):
        bins.append(pre_array[i:j,:])
        i = j
        j += bin_amount

    counts = []

    x = 5
    x_array = []
    for i in bins:
        total = np.sum(i[:,1])
        if total != 0:
            counts.append(total)
            x_array.append(x)
            x += 5
        
    return x_array, counts

def postrampspeed_processing(data):
    """ Does the PostRamp Speed processing to separate into bins based on 
    the data size. 

    Args: 
        data - individual dataframes

    Returns:
        x_array - the bin numbers
        counts - the counts for each speed
    
    """
    post_ramp_speed = data['PostRampSpeed']
    
    # Get counts for each speed
    post_counts = post_ramp_speed.value_counts()
    post_index = post_counts.index.to_numpy().reshape(-1,1)
    post_vals = post_counts.to_numpy().reshape(-1,1)
    
    post_array = np.hstack((post_index, post_vals))
    post_array = post_array[post_array[:,0].argsort()]

    # Binning into 5%
    bin_amount = int(np.ceil(len(post_array) * .05))
    max_range = int(np.floor(len(post_array) / bin_amount) + 1)
    bins = []
    i = 0
    j = bin_amount
    
    for z in range(0, max_range):
        bins.append(post_array[i:j,:])
        i = j
        j += bin_amount

    counts = []

    x = 5
    x_array = []
    for i in bins:
        total = np.sum(i[:,1])
        if total != 0:
            counts.append(total)
            x_array.append(x)
            x += 5
        
    return x_array, counts

############################# PreRamp Speed All Files #############################
plt.figure(figsize=(15,10))
for data in data_list:
    x_array, counts = prerampspeed_processing(data)
    plt.plot(x_array, counts)
       
plt.title("PreRamp Speed")
plt.xlabel("5% bins")
plt.ylabel("Counts")
plt.xticks(x_array)
plt.show()

############################# PostRamp Speed All Files #############################
plt.figure(figsize=(15,10))
for data in data_list:
    x_array, counts = postrampspeed_processing(data)
    plt.plot(x_array, counts)
    
       
plt.title("PostRamp Speed")
plt.xlabel("5% bins")
plt.ylabel("Counts")
plt.xticks(x_array)
plt.show()
