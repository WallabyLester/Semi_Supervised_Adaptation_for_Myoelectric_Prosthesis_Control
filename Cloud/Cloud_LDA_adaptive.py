"""
Contains a class for constructing, fitting, and predicting with an 
adaptive linear discriminant analysis (LDA) model. Specific for use 
with cloud data.

An LDA is often used for supervised classification problems. It 
estimates the probability that a new input belongs to every existing
class. The output class is the one with the highest probability. 
However, training an LDA requires overwriting the previous which 
doesn't account for past history in the case of EMG classification. 

Adaptation allows for the LDA to be adapted to new data instead of 
overwriting the previous classifier. Using the previous class means
and covariances, the LDA can be updated to account for past history
and the new data. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
import os
import glob

from Cloud_Rules import GameRules

def print_array(*args):
    """ Prints out entire array. 

        Params:
            *args - Any array like.
    """
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    pprint(*args)
    np.set_printoptions(**opt)

def load_data(subject):
    """ Reads in all virtual game files for one subject and saves those 
        for In the Zone.

        Params:
            subject - ID number to use.

        Returns:
            Gamedata - A list of Pandas dataframes for In the Zone data.
    """
    path = os.getcwd()
    csv_files = glob.glob(os.path.join(path, f"../Data/{subject}/VirtualGameData/VirtualArmGames_*"))
    csv_files = sorted(csv_files, key=lambda x:int(x[98:-4]))
    Gamedata = []
    print("Game data paths:")

    for i in csv_files:
        df = pd.read_csv(i)
        df = df[df['entryType'] == 0]
        # df = df[df['class'] != 0]
        if (df['gameName'] == "In the Zone").any():
            Gamedata.append(df)
            print(i)

    return Gamedata


class aLDA:
    """ Class for an adaptive linear discriminant analysis (LDA).

    Methods
    -------
    fit(X, y, classmeans=None, covariance=None, flag=0):
        Initializes and fits the LDA model.

    predict(X):
        Predicts on features using the fitted LDA model.

    updateMeanAndCov(meanMat, covMat, N, cur_feat):
        Updates the class means and covariances for adaptation.
    """
    def __init__(self):
        """ Constructs the necessary attributes for an LDA.
        """
        self.priors = None
        self.means = None
        self.cov = None
        self.covs = None
        self.classes = None

    def fit(self, X, y, classmeans=None, covariance=None, flag=0):
        """ Initializes and fits the LDA model.

            Params:
                X - The training features.
                y - The training labels.
                classmeans - Previous classmeans if adapting.
                covariance - Previous covariance if adapting.
                flag - Flag to indicate if adapting or if the first model.
                       (0 for first time)

            Returns:
                self.means - Class means.
                self.covs - Class covariances.
        """
        self.priors = dict()
        self.covs = dict()
        if flag == 0:
            self.means = dict()
            self.cov = np.cov(X, rowvar=False)
        else:
            self.means = classmeans
            self.cov = covariance
        
        self.classes = np.unique(y)

        for c in self.classes:
            self.priors[c] = 1 / len(self.classes)
            
            X_c = X[y == c]
            self.covs[c] = np.cov(X_c, rowvar=False)
            if flag == 0:
                self.means[c] = np.mean(X_c, axis=0)
        
        return (self.means, self.covs)

    def predict(self, X):
        """ Predicts on features using the fitted LDA model.
        
            Params:
                X - The test features.

            Returns:
                preds - Array of predictions.
        """
        preds = list()
        for x in X:
            posts = list()
            for c in self.classes:
                prior = np.log(self.priors[c])
                inv_cov = np.linalg.inv(self.cov)
                Wg = np.dot(inv_cov, self.means[c].T)
                Cg = -1/2 * self.means[c].dot(inv_cov).dot(self.means[c].T)
                likelihood = x.T @ Wg + Cg
                post = prior + likelihood
                posts.append(post)
            pred = self.classes[np.argmax(posts)]
            preds.append(pred)

        return np.array(preds)

    def updateMeanAndCov(self, meanMat, covMat, N, cur_feat):
        """ Updates the class means and covariances for adaptation.
        
            Params:
                meanMat - The previous class mean.
                covMat - The previous class covariance.
                N - The number of samples.
                cur_feat - The current feature.

            Returns:
                mean_feats - Adapted class mean.
                cov_updated - Adapted class covariance.
        """
        N = N+1
        ALPHA = N/(N+1)
        zero_mean_feats_prev = cur_feat - meanMat
        mean_feats = ALPHA*meanMat + (1-ALPHA) * cur_feat
        zero_mean_feats_cur = cur_feat - mean_feats
        point_cov = np.dot(zero_mean_feats_prev.T, zero_mean_feats_cur)
        # point_cov = np.array(point_cov,np.float64,order='F')
        cov_updated = ALPHA*covMat + (1-ALPHA) * point_cov

        return (mean_feats, cov_updated)

if __name__ == "__main__":
    """ Test cases for using the aLDA """
    # covariance = np.loadtxt("../Data/ID#31127011_2WProsthUse/VirtualGameData/Covariance-1_21_2021_18_26_12.csv", skiprows=1).reshape((56, 56))
    # covariance = covariance[0::7]
    # covariance = covariance[:, 0::7]
    # # print_array(covariance)

    # classmeans = pd.read_csv("../Data/ID#31127011_2WProsthUse/VirtualGameData/ClassMeans-1_21_2021_18_26_17.csv",header=None).to_numpy()
    # classes = classmeans[:,0]
    # print(classes)
    # classmeans = classmeans[:,1:]
    # classmeans = classmeans[:, 0::7]
    # # print_array(classmeans)
    # means_dict = {classes[0]:classmeans[0,:], classes[1]:classmeans[1,:], classes[2]:classmeans[2,:], classes[3]:classmeans[3,:], classes[4]:classmeans[4,:]}
    # means_dict = dict(list(means_dict.items())[1:-1])
    # print(means_dict)

    all_IDs = ["ID#31127011_2WProsthUse", "ID#31180011_2WProsthUse", "ID#32068222_2WProsthUse", 
               "ID#32098021_2WProsthUse", "ID#32132721_2WProsthUse_KW", "ID#32136722_2WProsthUse",
               "ID#32195432_2WProsthUse", "ID#51013322_2WProsthUse", "ID#51048532_2WProsthUse", 
               "ID#52054922_2WProsthUse"]

    """ Read in all In the Zone game data for one subject"""
    Gamedata = load_data("ID#52054922_2WProsthUse")
    print(f"Total games: {len(Gamedata)}")
    unadapted_err = []
    adapted_err = []

    """ First virtual game data"""
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

    """ Second virtual game data"""
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

    """Third virtual game data"""
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

    """Fourth virtual game data"""
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

    """Fifth virtual game data"""
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

    """Sixth virtual game data"""
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

    """Seventh virtual game data"""
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

    fig = plt.figure(figsize=(10, 5))
    plt.plot(unadapted_err, label='Unadapted')
    plt.plot(adapted_err, label='Adapted')
    plt.title("Error Rate Adapted vs. Unadapted")
    plt.ylabel("Error Rate")
    plt.xlabel("Games")
    plt.ylim(0, 100)
    plt.legend()
    plt.show()
