import numpy as np
import pandas as pd
from pprint import pprint

def print_array(*args):
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    pprint(*args)
    np.set_printoptions(**opt)

class LDA:
    def fit(self, X, y):
        self.priors = dict()
        self.means = dict()
        self.cov = np.cov(X, rowvar=False)

        self.classes = np.unique(y)

        for c in self.classes:
            X_c = X[y == c]
            self.priors[c] = X_c.shape[0] / X.shape[0]
            self.means[c] = np.mean(X_c, axis=0)

    def predict(self, X):
        preds = list()
        for x in X:
            posts = list()
            for c in self.classes:
                prior = np.log(self.priors[c])
                inv_cov = np.linalg.inv(self.cov)
                inv_cov_det = np.linalg.det(inv_cov)
                diff = x-self.means[c]
                likelihood = 0.5*np.log(inv_cov_det) - 0.5*diff.T @ inv_cov @ diff
                post = prior + likelihood
                posts.append(post)
            pred = self.classes[np.argmax(posts)]
            preds.append(pred)

        return np.array(preds)

if __name__ == "__main__":
    # update ID for specific patient and VirtualArmGames_ for which game session to use
    data = pd.read_csv("../Data/ID#31127011_2WProsthUse/VirtualGameData/VirtualArmGames_-1_21_2021_18_25_32.csv")
    class_data = data[['class', 'targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()
    class_data = class_data[~np.isnan(class_data).any(axis=1)]
    
    X = class_data[:, 2:]
    y = class_data[:, 1]

    lda = LDA()
    lda.fit(X, y)
    preds = lda.predict(X)

    # print_array(y)
    # print_array(preds)
    print(f"{np.around((y==preds).sum() / y.size, 2)}%")
    print(f"{np.around(((class_data[:,0]==y).sum() / y.size), 2)}")
