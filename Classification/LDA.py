import numpy as np
import pandas as pd

class LDA:
    def fit(self, X, y):
        self.priors = dict()
        self.means = dict()
        self.cov = np.cov(X, rowvar=False)
        print(self.cov)

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
    data = pd.read_csv("./Data/ID#/VirtualGameData/VirtualArmGames_.csv")
    class_data = data[['targetClass', 'emgChan1', 'emgChan2', 'emgChan3', 'emgChan4', 'emgChan5', 'emgChan6', 'emgChan7', 'emgChan8']].to_numpy()
    
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

    # X = data[:, 0:2]
    # y = data[:, 2]

    # lda = LDA()
    # lda.fit(X, y)
    # preds = lda.predict(X)

    # print(preds)