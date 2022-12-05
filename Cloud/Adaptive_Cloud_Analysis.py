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

# +
import pymysql
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Cloud_Rules import GameRules
from Cloud_LDA_adaptive import aLDA

# +
"engr_coaptengr"
db = pymysql.connect(host='34.135.162.65',user='coapt_admin',password="coaptengr",database='research_app')
cursor = db.cursor()
cursor.execute("SELECT * FROM in_the_zone_event_log_PLP")

df = pd.DataFrame(cursor.fetchall())
df = df[df[2] == 'CC0617']
df.head()

# +
# event id: 96
cursor.execute("SELECT * FROM in_the_zone_trial_log_PLP")

df = pd.DataFrame(cursor.fetchall(), columns=['trial_id', 'commandtime', 'in_the_zone_event_id', 'username', 'trial_num',
                                              'target_MID', 'timeout', 'zone_dwell_time', 'zone_target_speed', 
                                              'target_zone_tolerance', 'completion', 'completion_time', 'control_efficiency', 
                                              'num_overshoots', 'reaction_time'])
df = df[df['username'] == 'CC0617']
df = df[df['in_the_zone_event_id'] == 96]
trials = df[['trial_num', 'target_MID']].copy().to_numpy()
print(trials)

# +
cursor.execute("SELECT * FROM in_the_zone_data_log_PLP")

df = pd.DataFrame(cursor.fetchall(), columns=['commandnum', 'commandtime', 'in_the_zone_event_id', 'username', 'trial_num', 
                                              'play_time_elapsed', 'pre_ramp_speed', 'post_ramp_speed', 'smoothed_speed',
                                              'pre_ramp_MID', 'post_ramp_MID', 'smoothed_MID', 'move_timer', 'zone_fill',
                                              'state', 'mav1', 'mav2', 'mav3', 'mav4', 'mav5', 'mav6', 'mav7', 'mav8', 
                                              'lift_off_mask'])
df = df[df['username'] == 'CC0617']
df = df[df['in_the_zone_event_id'] == 96]
df = df[df['state'] != 0]
df.head()
# -
targetClass = []
for i in range(len(df)):
    targetClass.append(0)
df['targetClass'] = targetClass
df.head()

index = 0
for i in trials[:,0]:
    df.loc[df["trial_num"] == i, "targetClass"] = trials[index, 1]
    index += 1
df.head()

# +
adapted_err = []
adapted_wrules_err = []

class_data = df[['smoothed_MID', 'targetClass', 'mav1', 'mav2', 'mav3', 'mav4', 'mav5', 'mav6', 'mav7', 'mav8']].to_numpy()

""" Adapted """ 
X = adapt_sample[:, :-1]
y = adapt_sample[:, -1]

alda_norules = aLDA()
prev_means_norules, prev_covs_norules = alda_norules.fit(X, y, flag=0)

X = test_sample[:, :-1]
y = test_sample[:, -1]
alda_norules_preds = alda_norules.predict(X)

print(f"First game adapted: {np.around((y!=alda_norules_preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=alda_norules_preds).sum() / y.size, 2) * 100)

""" Adapted with rules """
rules = GameRules()
new_data = rules.augmented_rule(df)
new_data = np.vstack((new_data))[:,:-1]

X = new_data[:, 2:]
y = new_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

X = adapt_sample[:, :-1]
y = adapt_sample[:, -1]

alda = aLDA()
prev_means, prev_covs = alda.fit(X, y, flag=0)

X = test_sample[:, :-1]
y = test_sample[:, -1]
alda_preds = alda.predict(X)

# print_array(y)
# print_array(alda_preds)
print(f"First game adapted w/ rules: {np.around((y!=alda_preds).sum() / y.size, 2) * 100}%")

adapted_wrules_err.append(np.around((y!=alda_preds).sum() / y.size, 2) * 100)
# -


""" To create and upload a new table
import sqlite3

conn = sqlite3.connect('test_database') 
c = conn.cursor()

c.execute('''
          CREATE TABLE IF NOT EXISTS products
          ([product_id] INTEGER PRIMARY KEY, [product_name] TEXT, [price] INTEGER)
          ''')
          
c.execute('''
          INSERT INTO products (product_id, product_name, price)

                VALUES
                (1,'Computer',800),
                (2,'Printer',200),
                (3,'Tablet',300),
                (4,'Desk',450),
                (5,'Chair',150)
          ''')                     

conn.commit()
"""


