import numpy as np

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
    data = np.loadtxt("./test_data.csv", delimiter=",", skiprows=1)
    X = data[:, 0:2]
    y = data[:, 2]

    lda = LDA()
    lda.fit(X, y)
    preds = lda.predict(X)

    print(preds)