# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby
from collections import Counter
# from jupyterthemes import jtplot
# jtplot.reset()
import os
import glob

# +
""" Reads all data in depending on the data location and which ID#
Makes individual Pandas dataframes and places them in a 
python list: `data_list`
"""

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "../Data/ID#31127011_2WProsthUse/VirtualGameData/VirtualArmGames_*"))
data_list = []

for i in csv_files:
    df = pd.read_csv(i)
    data_list.append(df)

# +
# importing data from csv
single_set = pd.read_csv('../Data/ID#31127011_2WProsthUse/VirtualGameData/VirtualArmGames_-2_4_2021_16_34_19.csv')

# print(single_set.head())
emg_chan1 = single_set['emgChan1']
emg_chan2 = single_set['emgChan2']
emg_chan3 = single_set['emgChan3']
emg_chan4 = single_set['emgChan4']
emg_chan5 = single_set['emgChan5']
emg_chan6 = single_set['emgChan6']
emg_chan7 = single_set['emgChan7']
emg_chan8 = single_set['emgChan8']

print(f"Chan1 max: {emg_chan1.max()} Chan2 max: {emg_chan2.max()} Chan3 max: {emg_chan3.max()} Chan4 max: {emg_chan4.max()} Chan5 max: {emg_chan5.max()} Chan6 max: {emg_chan6.max()} Chan7 max: {emg_chan7.max()} Chan8 max: {emg_chan8.max()}")
print(f"Chan1 min: {emg_chan1.min()} Chan2 min: {emg_chan2.min()} Chan3 min: {emg_chan3.min()} Chan4 min: {emg_chan4.min()} Chan5 min: {emg_chan5.min()} Chan6 min: {emg_chan6.min()} Chan7 min: {emg_chan7.min()} Chan8 min: {emg_chan8.min()}")

# +
############################## Time Series Plots #############################
pre_ramp_speed = single_set['PreRampSpeed']
post_ramp_speed = single_set['PostRampSpeed']
class_targets = single_set['targetClass']
class_predictions = single_set['class']
points = len(pre_ramp_speed)
x = range(0, points, 1)
fig, ax = plt.subplots(2,2, sharex=True, figsize=(20,10))
y = pre_ramp_speed.to_numpy()
ax[0][0].plot(x, y)
ax[0][0].set_title("PreRamp Speed")
ax[0][0].set_ylabel("Speed")

y = post_ramp_speed.to_numpy()
ax[0][1].plot(x, y)
ax[0][1].set_title("PostRamp Speed")
ax[0][1].set_ylabel("Speed")

y = class_targets.to_numpy()
ax[1][0].plot(x, y)
ax[1][0].set_title("Motion Target")
ax[1][0].set_ylabel("Class")

y = class_predictions.to_numpy()
ax[1][1].plot(x, y)
ax[1][1].set_title("Motion Predictions")
ax[1][1].set_ylabel("Class")
plt.show()

# +
points = len(emg_chan1)
x = range(0, points, 1)
fig, ax = plt.subplots(4,2, sharex=True, figsize=(20,15))
y = emg_chan1
ax[0][0].plot(x, y)
ax[0][0].set_title("EMG Channel 1")
ax[0][0].set_ylabel("MAV")

y = emg_chan2
ax[0][1].plot(x, y)
ax[0][1].set_title("EMG Channel 2")
ax[0][1].set_ylabel("MAV")

y = emg_chan3
ax[1][0].plot(x, y)
ax[1][0].set_title("EMG Channel 3")
ax[1][0].set_ylabel("MAV")

y = emg_chan4
ax[1][1].plot(x, y)
ax[1][1].set_title("EMG Channel 4")
ax[1][1].set_ylabel("MAV")

y = emg_chan5
ax[2][0].plot(x, y)
ax[2][0].set_title("EMG Channel 5")
ax[2][0].set_ylabel("MAV")

y = emg_chan6
ax[2][1].plot(x, y)
ax[2][1].set_title("EMG Channel 6")
ax[2][1].set_ylabel("MAV")

y = emg_chan7
ax[3][0].plot(x, y)
ax[3][0].set_title("EMG Channel 7")
ax[3][0].set_ylabel("MAV")

y = emg_chan8
ax[3][1].plot(x, y)
ax[3][1].set_title("EMG Channel 8")
ax[3][1].set_ylabel("MAV")
plt.show()

# +
entry_type = single_set['entryType']
# # Normalized
# emg_chan1=(emg_chan1-emg_chan1.min())/(emg_chan1.max()-emg_chan1.min())
# emg_chan2=(emg_chan2-emg_chan2.min())/(emg_chan2.max()-emg_chan2.min())
# emg_chan3=(emg_chan3-emg_chan3.min())/(emg_chan3.max()-emg_chan3.min())
# emg_chan4=(emg_chan4-emg_chan4.min())/(emg_chan4.max()-emg_chan4.min())
# emg_chan5=(emg_chan5-emg_chan5.min())/(emg_chan5.max()-emg_chan5.min())
# emg_chan6=(emg_chan6-emg_chan6.min())/(emg_chan6.max()-emg_chan6.min())
# emg_chan7=(emg_chan7-emg_chan7.min())/(emg_chan7.max()-emg_chan7.min())
# emg_chan8=(emg_chan8-emg_chan8.min())/(emg_chan8.max()-emg_chan8.min())

# Ranging between 0 to 3
# New range
OldRange = (emg_chan1.max() - emg_chan1.min())
NewRange = (3 - 0)
emg_chan1_new = (((emg_chan1 - emg_chan1.min()) * NewRange) / OldRange) + 0
OldRange = (emg_chan2.max() - emg_chan2.min())
NewRange = (3 - 0)
emg_chan2_new = (((emg_chan2 - emg_chan2.min()) * NewRange) / OldRange) + 0
OldRange = (emg_chan3.max() - emg_chan3.min())
NewRange = (3 - 0)
emg_chan3_new = (((emg_chan3 - emg_chan3.min()) * NewRange) / OldRange) + 0
OldRange = (emg_chan4.max() - emg_chan4.min())
NewRange = (3 - 0)
emg_chan4_new = (((emg_chan4 - emg_chan4.min()) * NewRange) / OldRange) + 0
OldRange = (emg_chan5.max() - emg_chan5.min())
NewRange = (3 - 0)
emg_chan5_new = (((emg_chan5 - emg_chan5.min()) * NewRange) / OldRange) + 0
OldRange = (emg_chan6.max() - emg_chan6.min())
NewRange = (3 - 0)
emg_chan6_new = (((emg_chan6 - emg_chan6.min()) * NewRange) / OldRange) + 0
OldRange = (emg_chan7.max() - emg_chan7.min())
NewRange = (3 - 0)
emg_chan7_new = (((emg_chan7 - emg_chan7.min()) * NewRange) / OldRange) + 0
OldRange = (emg_chan8.max() - emg_chan8.min())
NewRange = (3 - 0)
emg_chan8_new = (((emg_chan8 - emg_chan8.min()) * NewRange) / OldRange) + 0

# Mean of MAVs ranging between 0 to 3
emg_sum = emg_chan1 + emg_chan2 + emg_chan3 + emg_chan4 + emg_chan5 + emg_chan6 + emg_chan7 + emg_chan8
emg_mean = emg_sum / 8
OldRange = (emg_mean.max() - emg_mean.min())
NewRange = (3 - 0)
emg_mean_new = (((emg_mean - emg_mean.min()) * NewRange) / OldRange) + 0

points = len(entry_type)
x = range(0, points, 1)
fig, ax = plt.subplots(3, sharex=True, figsize=(20,10))
y = entry_type
ax[0].plot(x, y)
ax[0].set_title("Entry Type")
ax[0].set_ylabel("Entry Number")

y = entry_type
ax[1].plot(x, y)
y = emg_chan1_new
ax[1].plot(x, y)
y = emg_chan2_new
ax[1].plot(x, y)
y = emg_chan3_new
ax[1].plot(x, y)
y = emg_chan4_new
ax[1].plot(x, y)
y = emg_chan5_new
ax[1].plot(x, y)
y = emg_chan6_new
ax[1].plot(x, y)
y = emg_chan7_new
ax[1].plot(x, y)
y = emg_chan8_new
ax[1].plot(x, y)
ax[1].set_title("Entry Type w/ EMG Outputs")
ax[1].set_ylabel("Entry Number + EMG Output")

y = entry_type
ax[2].plot(x, y)
y = emg_mean_new
ax[2].plot(x, y)
ax[2].set_title("Entry Type w/ Mean EMG Outputs")
ax[2].set_ylabel("Entry Number + Mean EMG Output")

plt.show()


# +
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
    pre_ramp_speed = pre_ramp_speed[pre_ramp_speed != 0]
    
    pre_ramp_speed = (pre_ramp_speed - pre_ramp_speed.min()) / (pre_ramp_speed.max() - pre_ramp_speed.min())
    # Separate games to separate plots (keep this one as well)
    # Last gamefile for each subject + mean of each patient
    # 1. do one with raw output and every single speed is a bin (with 0 too)
    # 2. do one with percents
    # use 0 to 254 (divide all by 255)
    
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
    post_ramp_speed = post_ramp_speed[post_ramp_speed != 0]
    
    post_ramp_speed = (post_ramp_speed - post_ramp_speed.min()) / (post_ramp_speed.max() - post_ramp_speed.min())
    
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


# +
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

# +
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

# +
############################## Consecutive Motion Plots ############################# 
test = single_set['class']
d = dict()

for k, v in groupby(test):
    d.setdefault(k, []).append(len(list(v)))

print(d)

new_list = []
for item in d:
    if item != 0:
        new_list.extend(d[item])

consecutive_motions = Counter(new_list)
print(new_list)
print(consecutive_motions)

x_list = []
greater_list = []
for key in consecutive_motions:
    if key <= 30:
        x_list.append(key)
        x_array = np.sort(x_list)
    else:
        greater_list.append(key)

counts = []
for item in x_array:
    counts.append(consecutive_motions[item])

greater_total = 0
for item in greater_list:
    greater_total += consecutive_motions[item]

x_array = np.append(x_array, 30)
counts.append(greater_total)

print(x_array)
print(counts)

plt.figure(figsize=(15,10))
plt.plot(x_array, counts)
plt.title("Consecutive Motions")
plt.xlabel("Consecutive Motions")
plt.ylabel("Counts")
plt.xticks(x_array)
plt.show()


# -

# Separate the two games here as well
# Plot out speed range for each consecutive motion 
# Where does 254 normally come up the most, how many consecutive motions needed to reach a max?
# Can we show moments where many consecutive motions occur then instance of a few in between, then see if it was misclassified or such?

def consecutivemotion_processing(data):
    """ Does the Consecutive Motions processing to separate into bins based on 
    the number of consecutive motions. After > 30, it sums the counts. 

    Args: 
        data - individual dataframes

    Returns:
        x_array - the number of consecutive motions
        counts - the counts for each number of motions
    """
    class_predictions = data['class']

    # makes a dictionary for consecutive motions of each class
    d = dict()
    for k, v in groupby(class_predictions):
        d.setdefault(k, []).append(len(list(v)))

    # make a list of consecutive motions without including 0 motion
    new_list = []
    for item in d:
        if item != 0:
            new_list.extend(d[item])

    # get the counts of each consecutive motion
    consecutive_motions = Counter(new_list)

    x_list = []
    greater_list = []
    for key in consecutive_motions:
        if key < 30:
            x_list.append(key)
            x_array = np.sort(x_list)
        else:
            greater_list.append(key)

    counts = []
    for item in x_array:
        counts.append(consecutive_motions[item])

    greater_total = 0
    for item in greater_list:
        greater_total += consecutive_motions[item]

    x_array = np.append(x_array, 30)
    counts.append(greater_total)

        
    return x_array, counts

# +
plt.figure(figsize=(15,10))
for data in data_list:
    x_array, counts = consecutivemotion_processing(data)
    plt.plot(x_array, counts)
    
plt.title("Consecutive Motions")
plt.xlabel("Consecutive Motions")
plt.ylabel("Counts")
plt.xticks(x_array)
plt.show()
# -


