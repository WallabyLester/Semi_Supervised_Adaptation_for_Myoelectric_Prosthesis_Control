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

# # Consecutive Motions For All Participants

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
""" Reads data in from all participants. The final virtual game data from 
In the Zone and Simon Says. Only takes data where the entry type = 0. 
python lists: 'InTheZone' and 'SimonSays'
"""

path = os.getcwd()
csv_files = glob.glob(os.path.join(path, "../Data/All_Participants/InTheZone/*"))
InTheZone = []

for i in csv_files:
    df = pd.read_csv(i)
    df = df[df['entryType'] == 0]
    InTheZone.append(df)
    
csv_files = glob.glob(os.path.join(path, "../Data/All_Participants/SimonSays/*"))
SimonSays = []

for i in csv_files:
    df = pd.read_csv(i)
    df = df[df['entryType'] == 0]
    SimonSays.append(df)


# -

def consecutive_motion_w_speed(motion_data):
    """ Finds the number of consecutive motions matched with the speed at each motion.
    
    Args: 
        motion_data - data set of form class and speed (numpy array)
        
    Returns:
        values_nozero - consecutive motions where the value is not equal to 0
    """
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

    values_nozero = []
    for value in values:
        if value[0,0] != 0:
            values_nozero.append(value)
            
    return values_nozero


# ## In the Zone

# ### Consecutive Motion with PostRamp Speed

# +
data = InTheZone[0]
motion_data = data[['class', 'PostRampSpeed']].to_numpy()

values_nozero = consecutive_motion_w_speed(motion_data)

plt.figure(figsize=(30,10))
for i in range(len(values_nozero)):
    if np.max(values_nozero[i][:, 1]) > 220:
        plt.plot(range(1, len(values_nozero[i])+1), values_nozero[i][:,1])
plt.title("Consecutive Motions In the Zone w/ PostRamp Speeds")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, 170, 2))
plt.yticks(range(0, 260, 10))
plt.show()

j = 0
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 1
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 2
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 3
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 4
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()
# -

# ### Consecutive Motion with PreRamp Speed

# +
data = InTheZone[0]
motion_data = data[['class', 'PreRampSpeed']].to_numpy()

values_nozero = consecutive_motion_w_speed(motion_data)

plt.figure(figsize=(30,10))
for i in range(len(values_nozero)):
    if np.max(values_nozero[i][:, 1]) > 220:
        plt.plot(range(1, len(values_nozero[i])+1), values_nozero[i][:,1])
plt.title("Consecutive Motions In the Zone w/ PreRamp Speeds")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, 170, 2))
plt.yticks(range(0, 260, 10))
plt.show()

j = 0
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 1
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 2
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 3
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 4
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions In the Zone w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()
# -

# ## Simon Says

# ### Consecutive Motion with PostRamp Speed

# +
data = SimonSays[0]
motion_data = data[['class', 'PostRampSpeed']].to_numpy()

values_nozero = consecutive_motion_w_speed(motion_data)

plt.figure(figsize=(30,10))
for i in range(len(values_nozero)):
    if np.max(values_nozero[i][:, 1]) > 220:
        plt.plot(range(1, len(values_nozero[i])+1), values_nozero[i][:,1])
plt.title("Consecutive Motions Simon Says w/ PostRamp Speeds")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, 34, 2))
plt.yticks(range(0, 260, 10))
plt.show()

j = 0
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 1
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 13
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 15
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 54
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PostRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PostRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()
# -

# ### Consecutive Motion with PreRamp Speed

# +
data = SimonSays[0]
motion_data = data[['class', 'PreRampSpeed']].to_numpy()

values_nozero = consecutive_motion_w_speed(motion_data)

plt.figure(figsize=(30,10))
for i in range(len(values_nozero)):
    if np.max(values_nozero[i][:, 1]) > 220:
        plt.plot(range(1, len(values_nozero[i])+1), values_nozero[i][:,1])
plt.title("Consecutive Motions Simon Says w/ PreRamp Speeds")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, 60, 2))
plt.yticks(range(0, 260, 10))
plt.show()

j = 0
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 1
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 13
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 15
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()

j = 54
plt.figure(figsize=(15,10))
plt.plot(range(1, len(values_nozero[j])+1), values_nozero[j][:,1])
plt.title(f"Consecutive Motions Simon Says w/ PreRamp Speeds for Class: {values_nozero[j][0,0]}")
plt.xlabel("Consecutive Motions")
plt.ylabel("PreRamp Speed")
plt.xticks(range(1, len(values_nozero[j])+1))
plt.yticks(range(0, 260, 10))
plt.show()
# -


