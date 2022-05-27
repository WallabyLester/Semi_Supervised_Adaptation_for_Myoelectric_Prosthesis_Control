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

# # Rules

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

# ## In the Zone

# ### Consecutive Motions 

# +
# using data from one user
data = InTheZone[0]
motion_data = data[['class', 'PostRampSpeed']].to_numpy()

# separate consecutive motions into separate lists in values
values = []
i_flag = 0
for i in range(1, len(motion_data)):
    if motion_data[i-1, 0] == motion_data[i, 0]:
        pass
    else:
        values.append(motion_data[i_flag:i, :])
        i_flag = i
        
    if i == len(motion_data)-1:
        values.append(motion_data[i_flag:i+1, :])

# remove class 0 if it does not follow or lead intended motion
if values[0][0,0] == 0:
    values.pop(0)
if values[-1][0,0] == 0:
    values.pop(-1)
values_nozero = values

# pull number of consecutive motions and last speed for the motions
consec_with_speed = np.zeros([len(values_nozero), 2])
i = 0
for value in values_nozero:
    consec_with_speed[i, 0] = len(value)
    consec_with_speed[i, 1] = value[-1,-1]
    i += 1

# get the average speed for each number of consecutive motions
consec_motions = np.unique(consec_with_speed[:,0])
consec_avg_speed = np.zeros([len(consec_motions), 2])
index = 0
for count in consec_motions:
    saved = []
    for element in consec_with_speed:
        if element[0] == count: 
            saved.append(element[1])
    avg = sum(saved)/len(saved)
    consec_avg_speed[index, 0] = count
    consec_avg_speed[index, 1] = avg
    index += 1
    
# using data from one user
data = InTheZone[0]
motion_data = data[['class', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

# separate consecutive motions into separate lists in values
values = []
i_flag = 0
for i in range(1, len(motion_data)):
    if motion_data[i-1, 0] == motion_data[i, 0]:
        pass
    else:
        values.append(motion_data[i_flag:i, :])
        i_flag = i
        
    if i == len(motion_data)-1:
        values.append(motion_data[i_flag:i+1, :])

# remove class 0 if it does not follow or lead intended motion
if values[0][0,0] == 0:
    values.pop(0)
if values[-1][0,0] == 0:
    values.pop(-1)
values_nozero = values

# pull number of consecutive motions and avg MAV RMS
consec_with_rms = np.zeros([len(values_nozero), 2])
i = 0
for value in values_nozero:
    x = 0
    consec_with_rms[i, 0] = len(value)
    consec_with_rms[i, 1] = np.sum([np.sqrt(np.mean((row[1:])**2)) for row in value]) / len(value)
    i += 1

# get the average RMS for each number of consecutive motions
consec_motions = np.unique(consec_with_rms[:,0])
consec_avg_rms = np.zeros([len(consec_motions), 2])
index = 0
for count in consec_motions:
    saved = []
    for element in consec_with_rms:
        if element[0] == count: 
            saved.append(element[1])
    avg = sum(saved)/len(saved)
    consec_avg_rms[index, 0] = count
    consec_avg_rms[index, 1] = avg
    index += 1

OldRange = (consec_avg_rms[:, 1].max() - consec_avg_rms[:, 1].min())
NewRange = (254 - 0)
consec_avg_rms[:, 1] = (((consec_avg_rms[:, 1] - consec_avg_rms[:, 1].min()) * NewRange) / OldRange) + 0

plt.figure(figsize=(30,10))
plt.bar(consec_avg_speed[:, 0], consec_avg_speed[:, 1], label='Avg Speed')
plt.plot(consec_avg_rms[:, 0], consec_avg_rms[:, 1], color='r', label='Avg MAV RMS')
plt.title("Consecutive Motions In the Zone w/ Avg PostRamp Speeds and Avg MAV RMS")
plt.xlabel("Consecutive Motions")
plt.ylabel("Avg PostRamp Speed and Avg MAV RMS")
plt.xticks(consec_avg_speed[:, 0], rotation=90, horizontalalignment='center')
plt.yticks(range(0, 260, 10))
# plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
plt.show()
# -

# ### Target vs Output Class

# +
oneZone = InTheZone[0]
predicted_class = oneZone['class'].to_numpy()
target_class = oneZone['targetClass'].to_numpy()

plt.figure(figsize=(30,10))
points = len(target_class)
x = range(points)
plt.plot(x, predicted_class, label='predicted class')
plt.plot(x, target_class, label='target class')
plt.title("Target vs. Predicted Class In the Zone")
plt.ylabel("Class")
plt.yticks(target_class)
plt.yticks(predicted_class)
plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
plt.show()

# +
oneZone = InTheZone[0]
class_data = oneZone[['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

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

# remove class 0 if it does not follow or lead intended motion
for row in values[0]:
    if row[0] == 0:
        values[0] = np.delete(values[0], 0, 0)
    else:
        break
    index += 1

for row in reversed(values[-1]):
    if row[0] == 0:
        values[-1] = np.delete(values[-1], -1, 0)
    else:
        break
    index += 1

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
    
# plot each set of target class with the predicted class
for value in values:
    plt.figure(figsize=(30,10))
    points = len(value)
    x = np.linspace(0, points, points)
    OldRange = (value[:,10].max() - value[:,10].min())
    NewRange = (value[:,0].max() - value[:,0].min())
    value[:,10] = (((value[:,10] - value[:,10].min()) * NewRange) / OldRange) + value[:,0].min()
    plt.plot(x, value[:,1], label='target class', color='g')
    plt.plot(x, value[:,0], label='predicted class', color='b')
    plt.plot(x, value[:,10], label='MAV RMS', color='r')
    plt.title(f"Target vs. Predicted Class In the Zone Target Class: {value[0,1]}")
    plt.ylabel("Class")
    plt.yticks(value[:,1])
    plt.yticks(value[:,0])
    plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
    plt.show()

# +
oneZone = InTheZone[0]
class_data = oneZone[['class', 'targetClass']].to_numpy()

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

# remove class 0 if it does not follow or lead intended motion
for row in values[0]:
    if row[0] == 0:
        values[0] = np.delete(values[0], 0, 0)
    else:
        break
    index += 1

for row in reversed(values[-1]):
    if row[0] == 0:
        values[-1] = np.delete(values[-1], -1, 0)
    else:
        break
    index += 1
    
# find number of consecutive motions broken up by predicted classes
for target in values:
# test = values[0]
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

    j = 0
    points = []
    for part in expanded_vals:
        if part[0,0] == part[0,1]:
            target_length = len(part)
            if j < len(expanded_vals)-1:
                next_length = len(expanded_vals[j+1])
                if expanded_vals[j+1][0,0] == 0:
                    value_flag = 0
                else:
                    value_flag = 1
                points.append([target_length, next_length, value_flag])
        j += 1
    
    if not points:
        continue
    print(points)
    points = np.vstack((points))

    plt.figure(figsize=(30,10))
    for row in points:
        if row[2] == 0:
            plt.plot(row[0], row[1], 'ro', markersize=10)
        else:
            plt.plot(row[0], row[1], 'go')
    plt.title(f"Consecutive Target vs. Predicted Class In the Zone Target Class: {target[0,1]}")
    plt.xlabel("Correct Class")
    plt.ylabel("Incorrect Class")
    plt.xticks(points[:,0])
    plt.yticks(points[:,1])
    legend_elements = [Line2D([0], [0], marker='o', color='g', label='Motion', markerfacecolor='g', markersize=15),
                       Line2D([0], [0], marker='o', color='r', label='No motion', markerfacecolor='r', markersize=15)]
    plt.legend(handles=legend_elements)
    plt.show()
# -

# ## Simon Says

# ### Consecutive Motions

# +
# using data from one user
data = SimonSays[0]
motion_data = data[['class', 'PostRampSpeed']].to_numpy()

# separate consecutive motions into separate lists in values
values = []
i_flag = 0
for i in range(1, len(motion_data)):
    if motion_data[i-1, 0] == motion_data[i, 0]:
        pass
    else:
        values.append(motion_data[i_flag:i, :])
        i_flag = i
        
    if i == len(motion_data)-1:
        values.append(motion_data[i_flag:i+1, :])

# remove class 0 if it does not follow or lead intended motion
if values[0][0,0] == 0:
    values.pop(0)
if values[-1][0,0] == 0:
    values.pop(-1)
values_nozero = values

# pull number of consecutive motions and last speed for the motions
consec_with_speed = np.zeros([len(values_nozero), 2])
i = 0
for value in values_nozero:
    consec_with_speed[i, 0] = len(value)
    consec_with_speed[i, 1] = value[-1,-1]
    i += 1

# get the average speed for each number of consecutive motions
consec_motions = np.unique(consec_with_speed[:,0])
consec_avg_speed = np.zeros([len(consec_motions), 2])
index = 0
for count in consec_motions:
    saved = []
    for element in consec_with_speed:
        if element[0] == count: 
            saved.append(element[1])
    avg = sum(saved)/len(saved)
    consec_avg_speed[index, 0] = count
    consec_avg_speed[index, 1] = avg
    index += 1

# using data from one user
data = SimonSays[0]
motion_data = data[['class', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

# separate consecutive motions into separate lists in values
values = []
i_flag = 0
for i in range(1, len(motion_data)):
    if motion_data[i-1, 0] == motion_data[i, 0]:
        pass
    else:
        values.append(motion_data[i_flag:i, :])
        i_flag = i
        
    if i == len(motion_data)-1:
        values.append(motion_data[i_flag:i+1, :])

# remove class 0 if it does not follow or lead intended motion
if values[0][0,0] == 0:
    values.pop(0)
if values[-1][0,0] == 0:
    values.pop(-1)
values_nozero = values

# pull number of consecutive motions and avg MAV RMS
consec_with_rms = np.zeros([len(values_nozero), 2])
i = 0
for value in values_nozero:
    x = 0
    consec_with_rms[i, 0] = len(value)
    consec_with_rms[i, 1] = np.sum([np.sqrt(np.mean((row[1:])**2)) for row in value]) / len(value)
    i += 1

# get the average RMS for each number of consecutive motions
consec_motions = np.unique(consec_with_rms[:,0])
consec_avg_rms = np.zeros([len(consec_motions), 2])
index = 0
for count in consec_motions:
    saved = []
    for element in consec_with_rms:
        if element[0] == count: 
            saved.append(element[1])
    avg = sum(saved)/len(saved)
    consec_avg_rms[index, 0] = count
    consec_avg_rms[index, 1] = avg
    index += 1

OldRange = (consec_avg_rms[:, 1].max() - consec_avg_rms[:, 1].min())
NewRange = (254 - 0)
consec_avg_rms[:, 1] = (((consec_avg_rms[:, 1] - consec_avg_rms[:, 1].min()) * NewRange) / OldRange) + 0

    
plt.figure(figsize=(30,10))
plt.bar(consec_avg_speed[:, 0], consec_avg_speed[:, 1])
plt.plot(consec_avg_rms[:, 0], consec_avg_rms[:, 1], color='r', label='Avg MAV RMS')
plt.title("Consecutive Motions Simon Says w/ Avg PostRamp Speeds and Avg MAV RMS")
plt.xlabel("Consecutive Motions")
plt.ylabel("Avg PostRamp Speed and Avg MAV RMS")
plt.xticks(consec_avg_speed[:, 0], rotation=90, horizontalalignment='center')
plt.yticks(range(0, 260, 10))
plt.show()

# +
oneZone = SimonSays[0]
predicted_class = oneZone['class'].to_numpy()
target_class = oneZone['targetClass'].to_numpy()

plt.figure(figsize=(30,10))
points = len(target_class)
x = range(points)
plt.plot(x, predicted_class, label='predicted class')
plt.plot(x, target_class, label='target class')
plt.title("Target vs. Predicted Class Simon Says")
plt.ylabel("Class")
plt.yticks(target_class)
plt.yticks(predicted_class)
plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
plt.show()

# +
oneZone = SimonSays[0]
class_data = oneZone[['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

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

# remove class 0 if it does not follow or lead intended motion
for row in values[0]:
    if row[0] == 0:
        values[0] = np.delete(values[0], 0, 0)
    else:
        break
    index += 1

for row in reversed(values[-1]):
    if row[0] == 0:
        values[-1] = np.delete(values[-1], -1, 0)
    else:
        break
    index += 1

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
    
# plot each set of target class with the predicted class
for value in values:
    plt.figure(figsize=(30,10))
    points = len(value)
    x = np.linspace(0, points, points)
    OldRange = (value[:,10].max() - value[:,10].min())
    NewRange = (value[:,0].max() - value[:,0].min())
    value[:,10] = (((value[:,10] - value[:,10].min()) * NewRange) / OldRange) + value[:,0].min()
    plt.plot(x, value[:,1], label='target class', color='g')
    plt.plot(x, value[:,0], label='predicted class', color='b')
    plt.plot(x, value[:,10], label='MAV RMS', color='r')
    plt.title(f"Target vs. Predicted Class Simon Says Target Class: {value[0,1]}")
    plt.ylabel("Class")
    plt.yticks(value[:,1])
    plt.yticks(value[:,0])
    plt.legend(bbox_to_anchor=(1, 1),loc='upper left')
    plt.show()

# +
oneZone = SimonSays[0]
class_data = oneZone[['class', 'targetClass']].to_numpy()

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

# remove class 0 if it does not follow or lead intended motion
for row in values[0]:
    if row[0] == 0:
        values[0] = np.delete(values[0], 0, 0)
    else:
        break
    index += 1

for row in reversed(values[-1]):
    if row[0] == 0:
        values[-1] = np.delete(values[-1], -1, 0)
    else:
        break
    index += 1
    
# find number of consecutive motions broken up by predicted classes
for target in values:
# test = values[0]
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

    j = 0
    points = []
    for part in expanded_vals:
        if part[0,0] == part[0,1]:
            target_length = len(part)
            if j < len(expanded_vals)-1:
                next_length = len(expanded_vals[j+1])
                if expanded_vals[j+1][0,0] == 0:
                    value_flag = 0
                else:
                    value_flag = 1
                points.append([target_length, next_length, value_flag])
        j += 1
    
    if not points:
        continue
    print(points)
    points = np.vstack((points))

    plt.figure(figsize=(30,10))
    for row in points:
        if row[2] == 0:
            plt.plot(row[0], row[1], 'ro', markersize=10)
        else:
            plt.plot(row[0], row[1], 'go')
    plt.title(f"Consecutive Target vs. Predicted Class Simon Says Target Class: {target[0,1]}")
    plt.xlabel("Correct Class")
    plt.ylabel("Incorrect Class")
    plt.xticks(points[:,0])
    plt.yticks(points[:,1])
    legend_elements = [Line2D([0], [0], marker='o', color='g', label='Motion', markerfacecolor='g', markersize=15),
                       Line2D([0], [0], marker='o', color='r', label='No motion', markerfacecolor='r', markersize=15)]
    plt.legend(handles=legend_elements)
    plt.show()
# -


