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

# # Rule Testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from itertools import groupby
from collections import Counter
# from jupyterthemes import jtplot
# jtplot.reset()
import os
import glob
from copy import deepcopy

# +
""" Reads data in from all participants. The final virtual game data from 
In the Zone and Simon Says. Only takes data where the entry type = 0. 
python lists: 'InTheZone' and 'SimonSays'
"""

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "../Data/All_Participants/InTheZone/*"))
InTheZone = []
print("In the Zone Paths:")

for i in csv_files:
    df = pd.read_csv(i)
    df = df[df['entryType'] == 0]
    InTheZone.append(df)
    print(i)
    
csv_files = glob.glob(os.path.join(path, "../Data/All_Participants/SimonSays/*"))
SimonSays = []
print("Simon Says Paths:")

for i in csv_files:
    df = pd.read_csv(i)
    df = df[df['entryType'] == 0]
    SimonSays.append(df)
    print(i)


# -

# ## Base Rule
# 1. Base rule: 
#
#     if the number of consecutive motions == target > 4:
#     
#     if the next set of motions < 4 and the motion after == target:
#     
#     relabel the data
#

def base_rule(data):
    """ Finds the number of points to relabel based on the base rule.
    
    Uses a consecutive motion threshold of 4. 
    
    Args: 
        data - the virtual game data
        
    Returns:
        total_changed_sum - the total number of data points relabeled
        percentage_changed - the percentage of data points relabeled
    """
    class_data = data[['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

    # separate into lists based on target class
    values = []
    i_flag = 0
    for i in range(1, len(class_data)):
        if class_data[i-1, 1] == class_data[i, 1]:
            pass
        else:
            values.append(class_data[i_flag:i, :])
            i_flag = i

        if i == len(class_data)-1:
            values.append(class_data[i_flag:i+1, :])

    # find RMS of MAVs at each time point
    j = 0
    for value in values:
        rms = np.zeros((len(value), 1))
        i = 0
        for row in value:
            rms[i] = np.sqrt(np.mean((row[2:])**2))
            i += 1
        value = np.hstack((value, rms))
        values[j] = value
        j += 1

    # find number of points to relabel from base rule
    total_changed = []
    for target in values:    
        # separate target by predicted classes
        expanded_vals = []
        i_flag = 0
        for i in range(1, len(target)):
            if target[i-1, 0] == target[i, 0]:
                pass
            else:
                expanded_vals.append(target[i_flag:i, :])
                i_flag = i

            if i == len(target)-1:
                expanded_vals.append(target[i_flag:i+1, :])

        # get the counts of consecutive motions of predicted classes vs target
        expanded_vals_copy = deepcopy(expanded_vals)
        j = 0
        points = []
        changed = 0
        for part in expanded_vals:
            # check if predicted == target
            if part[0,0] == part[0,1]:
                target_length = len(part)
                # boundary check to make sure not exceeding array size
                if j < len(expanded_vals)-2:
                    next_length = len(expanded_vals[j+1])
                    nextnext_length = len(expanded_vals[j+2])
                    # check if motions after == target
                    if expanded_vals[j+2][0,0] == part[0,1]:
                        points.append([target_length, next_length, nextnext_length])
                        # implement < 4 rule
                        if next_length < 4:
                            expanded_vals_copy[j+1][:,0] = part[0,1]
                            changed += len(expanded_vals_copy[j+1][:,0])
            j += 1

        if not points:
            continue
        print(f"Consecutive motions [target, incorect, target]: \n{points}")
        print(f"Number of relabeled points: {changed}")
        total_changed.append(changed)
        
    total_changed_sum = np.sum(total_changed)
    percentage_changed = np.around((np.sum(total_changed)/len(class_data))*100, 2)
    
    return total_changed_sum, percentage_changed


data = InTheZone[1]
total_changed_sum, percentage_changed = base_rule(data)
print(f"Total changed: {total_changed_sum}")
print(f"Percentage changed: {percentage_changed}%")


# ## Augmented Rule
# 2. Augmented rule:<br>
#     Include MAV RMS
#     
#     **For no motion**<br>
#     if the number of consecutive motions == target > 4:<br>
#     if the next set of motions == 0 and the motion after == target and MAV RMS falls to either a baseline (stored from no motion classifications) or under a threshold:<br>
#     don't include the data or label as no motion class
#     
#     **For motion**<br>
#     if the number of consecutive motions == target > 4:<br>
#     if the next set of motions < 4 and the motion after == target and MAV RMS is within some range +/- epsilon:<br>
#     relabel the data<br>
#     if the next set of motions < 4 and motion after == target, but MAV RMS spiked during incorrect motion then converged:<br>
#     label that data as the motion class that was predicted or do nothing

def augmented_rule(data):
    """ Finds the number of points to relabel based on the augmented rule.
    
    Uses a consecutive motion threshold of 4 while bringing MAV RMS into 
    account.
    
    Args: 
        data - the virtual game data
        
    Returns:
        total_changed_sum - the total number of data points relabeled
        percentage_changed - the percentage of data points relabeled
    """
    class_data = data[['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

    # separate into lists based on target class
    values = []
    i_flag = 0
    for i in range(1, len(class_data)):
        if class_data[i-1, 1] == class_data[i, 1]:
            pass
        else:
            values.append(class_data[i_flag:i, :])
            i_flag = i

        if i == len(class_data)-1:
            values.append(class_data[i_flag:i+1, :])

    # find RMS of MAVs at each time point
    j = 0
    for value in values:
        rms = np.zeros((len(value), 1))
        i = 0
        for row in value:
            rms[i] = np.sqrt(np.mean((row[2:])**2))
            i += 1
        value = np.hstack((value, rms))
        values[j] = value
        j += 1

    # find number of points to relabel
    total_changed = []
    for target in values:    
        # separate target by predicted classes
        expanded_vals = []
        i_flag = 0
        for i in range(1, len(target)):
            if target[i-1, 0] == target[i, 0]:
                pass
            else:
                expanded_vals.append(target[i_flag:i, :])
                i_flag = i

            if i == len(target)-1:
                expanded_vals.append(target[i_flag:i+1, :])

        # get the counts of consecutive motions of predicted classes vs target
        expanded_vals_copy = deepcopy(expanded_vals)
        j = 0
        points = []
        changed = 0
        no_motion = 0
        for part in expanded_vals:
            # check if predicted == target
            if part[0,0] == part[0,1]:
                target_length = len(part)
                # boundary check to make sure not exceeding array size
                if j < len(expanded_vals)-2:
                    next_length = len(expanded_vals[j+1])
                    nextnext_length = len(expanded_vals[j+2])
                    # check if motions after == target
                    if expanded_vals[j+2][0,0] == part[0,1]:
                        points.append([target_length, next_length, nextnext_length])
                        # implement rules
                        if next_length < 4:
                            # no motion relabeling
                            if expanded_vals[j+1][0,0] == 0:
                                value_flag = 0
                                # if RMS goes lower by 500 then relabel as no motion otherwise relabel as motion target
                                RMS_curr = np.sqrt(np.mean((expanded_vals[j][:,-1])**2))
                                RMS_next = np.sqrt(np.mean((expanded_vals[j+1][:,-1])**2))
                                RMS_nextnext = np.sqrt(np.mean((expanded_vals[j+2][:,-1])**2))
                                if RMS_next <= ((RMS_curr-500) and (RMS_nextnext-500)):
                                    expanded_vals_copy[j+1][:,1] = 0
                                    no_motion += len(expanded_vals_copy[j+1][:,1])
                                else:
                                    expanded_vals_copy[j+1][:,0] = part[0,1]
                                    changed += len(expanded_vals_copy[j+1][:,0])
                            # motion relabeling
                            else:
                                expanded_vals_copy[j+1][:,0] = part[0,1]
                                changed += len(expanded_vals_copy[j+1][:,0])
            j += 1

        if not points:
            continue
        print(f"Consecutive motions [target, incorect, target]: \n{points}")
        print(f"Number of relabeled points: {changed+no_motion}")
        print(f"Number of no motion relabels: {no_motion}")
        total_changed.append(changed+no_motion)
        
    total_changed_sum = np.sum(total_changed)
    percentage_changed = np.around((np.sum(total_changed)/len(class_data))*100, 2)
    
    return total_changed_sum, percentage_changed


data = InTheZone[1]
total_changed_sum, percentage_changed = augmented_rule(data)
print(f"Total changed: {total_changed_sum}")
print(f"Percentage changed: {percentage_changed}%")


