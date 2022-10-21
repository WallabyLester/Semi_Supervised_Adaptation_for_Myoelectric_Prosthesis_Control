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

# # Adaptive LDA Base Rule

# +
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import os
import glob

from Rules import GameRules
from LDA_adaptive import aLDA
from LDA_adaptive import print_array, load_data
# -

# ## Base Rule

# +
all_IDs = ["ID#31127011_2WProsthUse", "ID#31180011_2WProsthUse", "ID#32068222_2WProsthUse", 
               "ID#32098021_2WProsthUse", "ID#32132721_2WProsthUse_KW", "ID#32136722_2WProsthUse",
               "ID#32195432_2WProsthUse", "ID#51013322_2WProsthUse", "ID#51048532_2WProsthUse", 
               "ID#52054922_2WProsthUse"]

""" Read in all In the Zone game data for one subject"""
Gamedata = load_data("ID#52054922_2WProsthUse")
print(f"Total games: {len(Gamedata)}")
# -

# First virtual game data

# +
unadapted_err = []
adapted_err = []

class_data = Gamedata[0][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

X = adapt_sample[:, :-1]
y = adapt_sample[:, -1]

lda = aLDA()
_, _ = lda.fit(X, y, flag=0)

X = test_sample[:, :-1]
y = test_sample[:, -1]
lda_preds = lda.predict(X)

# print_array(y)
# print_array(lda_preds)

print("Error Rates")
print(f"First game unadapted: {np.around((y!=lda_preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=lda_preds).sum() / y.size, 2) * 100)

rules = GameRules()
new_data = rules.base_rule(Gamedata[0])
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
print(f"First game w/ rules: {np.around((y!=alda_preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=alda_preds).sum() / y.size, 2) * 100)
# -

# Second virtual game data

# +
class_data = Gamedata[1][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

"""Test without adapting"""
X = test_sample[:, :-1]
y = test_sample[:, -1]

preds = lda.predict(X)
print(f"Second game unadapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

"""Test with adaptation"""
rules = GameRules()
new_data = rules.base_rule(Gamedata[1])
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
    # N = len(means[key])
    N = len(classes)
    cur_feat = means[key]
    prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

temp_covs = temp_covs / len(classes)

prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

X = test_sample[:, :-1]
y = test_sample[:, -1]
preds = alda.predict(X)
print(f"Second game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

# -

# Third virtual game data

# +
class_data = Gamedata[2][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

"""Test without adapting"""
X = test_sample[:, :-1]
y = test_sample[:, -1]

preds = lda.predict(X)
print(f"Third game unadapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

"""Test with adaptation"""
rules = GameRules()
new_data = rules.base_rule(Gamedata[2])
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
    # N = len(means[key])
    N = len(classes)
    cur_feat = means[key]
    prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

temp_covs = temp_covs / len(classes)

prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

X = test_sample[:, :-1]
y = test_sample[:, -1]
preds = alda.predict(X)
print(f"Third game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

# -

# Fourth virtual game data

# +
class_data = Gamedata[3][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

"""Test without adapting"""
X = test_sample[:, :-1]
y = test_sample[:, -1]

preds = lda.predict(X)
print(f"Fourth game unadapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

"""Test with adaptation"""
rules = GameRules()
new_data = rules.base_rule(Gamedata[3])
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
    # N = len(means[key])
    N = len(classes)
    cur_feat = means[key]
    prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

temp_covs = temp_covs / len(classes)

prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

X = test_sample[:, :-1]
y = test_sample[:, -1]
preds = alda.predict(X)
print(f"Fourth game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

# -

# Fifth virtual game data

# +
class_data = Gamedata[4][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

"""Test without adapting"""
X = test_sample[:, :-1]
y = test_sample[:, -1]

preds = lda.predict(X)
print(f"Fifth game unadapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

"""Test with adaptation"""
rules = GameRules()
new_data = rules.base_rule(Gamedata[4])
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
    # N = len(means[key])
    N = len(classes)
    cur_feat = means[key]
    prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

temp_covs = temp_covs / len(classes)

prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

X = test_sample[:, :-1]
y = test_sample[:, -1]
preds = alda.predict(X)
print(f"Fifth game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)
# -

# Sixth virtual game data

# +
class_data = Gamedata[5][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

"""Test without adapting"""
X = test_sample[:, :-1]
y = test_sample[:, -1]

preds = lda.predict(X)
print(f"Sixth game unadapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

"""Test with adaptation"""
rules = GameRules()
new_data = rules.base_rule(Gamedata[5])
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
    # N = len(means[key])
    N = len(classes)
    cur_feat = means[key]
    prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

temp_covs = temp_covs / len(classes)

prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

X = test_sample[:, :-1]
y = test_sample[:, -1]
preds = alda.predict(X)
print(f"Sixth game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)
# -

# Seventh virtual game data

# +
class_data = Gamedata[6][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

"""Test without adapting"""
X = test_sample[:, :-1]
y = test_sample[:, -1]

preds = lda.predict(X)
print(f"Seventh game unadapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

"""Test with adaptation"""
rules = GameRules()
new_data = rules.base_rule(Gamedata[6])
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
    # N = len(means[key])
    N = len(classes)
    cur_feat = means[key]
    prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

temp_covs = temp_covs / len(classes)

prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

X = test_sample[:, :-1]
y = test_sample[:, -1]
preds = alda.predict(X)
print(f"Seventh game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)
# -

# Eighth virtual game data

# +
class_data = Gamedata[7][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

"""Test without adapting"""
X = test_sample[:, :-1]
y = test_sample[:, -1]

preds = lda.predict(X)
print(f"Eighth game unadapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

"""Test with adaptation"""
rules = GameRules()
new_data = rules.base_rule(Gamedata[6])
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
    # N = len(means[key])
    N = len(classes)
    cur_feat = means[key]
    prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

temp_covs = temp_covs / len(classes)

prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

X = test_sample[:, :-1]
y = test_sample[:, -1]
preds = alda.predict(X)
print(f"Eighth game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)
# -

# Ninth virtual game data

# +
class_data = Gamedata[8][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

"""Test without adapting"""
X = test_sample[:, :-1]
y = test_sample[:, -1]

preds = lda.predict(X)
print(f"Ninth game unadapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

"""Test with adaptation"""
rules = GameRules()
new_data = rules.base_rule(Gamedata[6])
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
    # N = len(means[key])
    N = len(classes)
    cur_feat = means[key]
    prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

temp_covs = temp_covs / len(classes)

prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

X = test_sample[:, :-1]
y = test_sample[:, -1]
preds = alda.predict(X)
print(f"Ninth game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)
# -

# Tenth virtual game data

# +
class_data = Gamedata[9][['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()

X = class_data[:, 2:]
y = class_data[:, 1]

np.random.seed(42)
combined = np.hstack((X,y.reshape(len(y),1)))
np.random.shuffle(combined)
adapt_sample = combined[:len(y)//2, :]
test_sample = combined[len(y)//2:, :]

"""Test without adapting"""
X = test_sample[:, :-1]
y = test_sample[:, -1]

preds = lda.predict(X)
print(f"Tenth game unadapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

unadapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)

"""Test with adaptation"""
rules = GameRules()
new_data = rules.base_rule(Gamedata[6])
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
    # N = len(means[key])
    N = len(classes)
    cur_feat = means[key]
    prev_means[key], prev_covs[key] = alda.updateMeanAndCov(meanMat, covMat, N, cur_feat)

temp_covs = temp_covs / len(classes)

prev_means, prev_covs = alda.fit(X, y, classmeans=prev_means, covariance=temp_covs, flag=1)

X = test_sample[:, :-1]
y = test_sample[:, -1]
preds = alda.predict(X)
print(f"Tenth game adapted: {np.around((y!=preds).sum() / y.size, 2) * 100}%")

adapted_err.append(np.around((y!=preds).sum() / y.size, 2) * 100)
# -

fig = plt.figure(figsize=(30, 10))
plt.plot(unadapted_err, label='Unadapted', marker='X', markersize=10)
plt.plot(adapted_err, label='Adapted', marker='X', markersize=10)
plt.title("Error Rate Adapted vs. Unadapted")
plt.ylabel("Error Rate")
plt.xlabel("Games")
plt.yticks(np.arange(0, 110, 10))
plt.legend()
plt.show()


