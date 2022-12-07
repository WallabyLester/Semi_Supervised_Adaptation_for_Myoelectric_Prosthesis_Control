"""
Integration of adaptive LDA and adaptive LDA with rules and cloud based storage.

Runs in the cloud to use calibration and virtual game data. If a previous
classifier exists it will query the database to pull in the classifier. If 
a previous classifier has not been made it will treat the data as the first
game and create a new LDA. 

Logins for cloud database have been removed for privacy. Data is not included.
"""

import pymysql
import numpy as np
import pandas as pd
import argparse

from Cloud_Rules import GameRules
from Cloud_LDA_adaptive import aLDA

parser = argparse.ArgumentParser()
 
parser.add_argument("-u", "--username", help="Pass the desired username")
parser.add_argument("-e", "--eventid", help="Pass the desired event id")
 
args = parser.parse_args()
# args.username
# args.eventid
print(f"\n {args.eventid}")

# info not included
db = pymysql.connect(host=,user=,password=,database=)
cursor = db.cursor()

pass_flag = False
try:
    sql = "SELECT MIDs FROM test_blob_classifier WHERE username = %s"
    val = "CC004"
    cursor.execute(sql,(val,))
    results = cursor.fetchone()
    print("There is a previous classifier")
    pass_flag = True
except:
    print("There is not a previous classifier")
    pass

if pass_flag:
    # pull in old classifier 
    # query classifier data inserted above from database
    sql = "SELECT MIDs FROM test_blob_classifier WHERE username = %s"
    val = "CC004"
    cursor.execute(sql,(val,)) 
    results = cursor.fetchone()
    print("Array using fromstring():")
    qMIDs = np.frombuffer(results[0])
    print(qMIDs)

    sql = "SELECT MAVs FROM test_blob_classifier WHERE username = %s"
    cursor.execute(sql,(val,)) 
    results = cursor.fetchone()
    print("Array using fromstring():")
    qMAVs = np.frombuffer(results[0])
    print(qMAVs)

    sql = "SELECT pooledCov FROM test_blob_classifier WHERE username = %s"
    cursor.execute(sql,(val,)) 
    results = cursor.fetchone()
    print("Array using fromstring():")
    qpooledCov = np.frombuffer(results[0])
    print(qpooledCov)

    # pull in data
    cursor.execute("SELECT * FROM in_the_zone_event_log_PLP")

    df = pd.DataFrame(cursor.fetchall())
    df = df[df[2] == val]

    cursor.execute("SELECT * FROM in_the_zone_trial_log_PLP")

    df = pd.DataFrame(cursor.fetchall(), columns=['trial_id', 'commandtime', 'in_the_zone_event_id', 'username', 'trial_num',
                                                'target_MID', 'timeout', 'zone_dwell_time', 'zone_target_speed', 
                                                'target_zone_tolerance', 'completion', 'completion_time', 'control_efficiency', 
                                                'num_overshoots', 'reaction_time'])
    df = df[df['username'] == val]
    df = df[df['in_the_zone_event_id'] == 96]   # choose event id
    trials = df[['trial_num', 'target_MID']].copy().to_numpy()

    cursor.execute("SELECT * FROM in_the_zone_data_log_PLP")

    df = pd.DataFrame(cursor.fetchall(), columns=['commandnum', 'commandtime', 'in_the_zone_event_id', 'username', 'trial_num', 
                                                'play_time_elapsed', 'pre_ramp_speed', 'post_ramp_speed', 'smoothed_speed',
                                                'pre_ramp_MID', 'post_ramp_MID', 'smoothed_MID', 'move_timer', 'zone_fill',
                                                'state', 'mav1', 'mav2', 'mav3', 'mav4', 'mav5', 'mav6', 'mav7', 'mav8', 
                                                'lift_off_mask'])
    df = df[df['username'] == val]
    df = df[df['in_the_zone_event_id'] == 96]   # choose event id
    df = df[df['state'] != 0]  

    targetClass = []
    for i in range(len(df)):
        targetClass.append(0)
    df['targetClass'] = targetClass

    index = 0
    for i in trials[:,0]:
        df.loc[df["trial_num"] == i, "targetClass"] = trials[index, 1]
        index += 1

    adapted_err = []
    adapted_wrules_err = []

    # Start adaptation
    class_data = df[['smoothed_MID', 'targetClass', 'mav1', 'mav2', 'mav3', 'mav4', 'mav5', 'mav6', 'mav7', 'mav8']].to_numpy()

    X = class_data[:, 2:]
    y = class_data[:, 1]

    np.random.seed(42)
    combined = np.hstack((X,y.reshape(len(y),1)))
    np.random.shuffle(combined)
    adapt_sample = combined[:len(y)//2, :]
    test_sample = combined[len(y)//2:, :]

    """ Test with adaptation """
    alda_norules = aLDA()
    X = adapt_sample[:, :-1]
    y = adapt_sample[:, -1]

    classes = np.unique(y)
    means = dict()
    covs = dict()

    for c in classes:
        X_c = X[y == c]
        means[c] = np.mean(X_c, axis=0)
        covs[c] = np.cov(X_c, rowvar=False)

    temp_covs = np.zeros((8,8))
    for key in prev_means_norules:
        meanMat = prev_means_norules[key]
        covMat = prev_covs_norules[key]
        temp_covs += covMat
        N = len(means[key])
    #     N = len(classes)
        cur_feat = means[key]
        prev_means_norules[key], prev_covs_norules[key] = alda_norules.updateMeanAndCov(meanMat, covMat, N, cur_feat)

    temp_covs = temp_covs / len(classes)

    prev_means_norules, prev_covs_norules = alda_norules.fit(X, y, classmeans=prev_means_norules, covariance=prev_covs_norules, flag=1)

    X = test_sample[:, :-1]
    y = test_sample[:, -1]
    preds = alda_norules.predict(X)
    print(f"Second game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

    adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

    """Test with adaptation and rules"""
    alda = aLDA()
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

    classes = np.unique(y)
    means = dict()
    covs = dict()

    for c in classes:
        X_c = X[y == c]
        means[c] = np.mean(X_c, axis=0)
        covs[c] = np.cov(X_c, rowvar=False)

    temp_covs = np.zeros((8,8))
    for key in prev_means:
        meanMat = prev_means[key]
        covMat = prev_covs[key]
        temp_covs += covMat
        N = len(means[key])
    #     N = len(classes)
        cur_feat = means[key]
        prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

    temp_covs = temp_covs / len(classes)

    prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

    X = test_sample[:, :-1]
    y = test_sample[:, -1]
    preds = alda.predict(X)
    print(f"Second game adapted w/ rules: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

    adapted_wrules_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)
else:
    val = "CC004"
    cursor.execute("SELECT * FROM in_the_zone_event_log_PLP")

    df = pd.DataFrame(cursor.fetchall())
    df = df[df[2] == val]

    cursor.execute("SELECT * FROM in_the_zone_trial_log_PLP")

    df = pd.DataFrame(cursor.fetchall(), columns=['trial_id', 'commandtime', 'in_the_zone_event_id', 'username', 'trial_num',
                                                'target_MID', 'timeout', 'zone_dwell_time', 'zone_target_speed', 
                                                'target_zone_tolerance', 'completion', 'completion_time', 'control_efficiency', 
                                                'num_overshoots', 'reaction_time'])
    df = df[df['username'] == val]
    df = df[df['in_the_zone_event_id'] == 96]   # choose event id
    trials = df[['trial_num', 'target_MID']].copy().to_numpy()

    cursor.execute("SELECT * FROM in_the_zone_data_log_PLP")

    df = pd.DataFrame(cursor.fetchall(), columns=['commandnum', 'commandtime', 'in_the_zone_event_id', 'username', 'trial_num', 
                                                'play_time_elapsed', 'pre_ramp_speed', 'post_ramp_speed', 'smoothed_speed',
                                                'pre_ramp_MID', 'post_ramp_MID', 'smoothed_MID', 'move_timer', 'zone_fill',
                                                'state', 'mav1', 'mav2', 'mav3', 'mav4', 'mav5', 'mav6', 'mav7', 'mav8', 
                                                'lift_off_mask'])
    df = df[df['username'] == val]
    df = df[df['in_the_zone_event_id'] == 96]   # choose event id
    df = df[df['state'] != 0]   

    targetClass = []
    for i in range(len(df)):
        targetClass.append(0)
    df['targetClass'] = targetClass

    index = 0
    for i in trials[:,0]:
        df.loc[df["trial_num"] == i, "targetClass"] = trials[index, 1]
        index += 1

    adapted_err = []
    adapted_wrules_err = []

    class_data = df[['smoothed_MID', 'targetClass', 'mav1', 'mav2', 'mav3', 'mav4', 'mav5', 'mav6', 'mav7', 'mav8']].to_numpy()

    """ Adapted """ 
    X = class_data[:, 2:]
    y = class_data[:, 1]

    np.random.seed(42)
    combined = np.hstack((X,y.reshape(len(y),1)))
    np.random.shuffle(combined)
    adapt_sample = combined[:len(y)//2, :]
    test_sample = combined[len(y)//2:, :]

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
