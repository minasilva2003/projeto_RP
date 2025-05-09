import numpy as np
from sklearn import mixture
from scipy.stats import multivariate_normal

class BayesianGaussianClassifier:
    def __init__(self, data_info):
        self.classifier = None
        self.classifier_label = "Bayes_Classifier"
        self.data_info = data_info

    def train(self, X_train, Y_train):
        """
        Fit the Bayesian classifier using Gaussian models for each class.
        Assumes binary classification with labels 1 and 2.
        """
        classes = np.unique(Y_train)
        if len(classes) != 2 or not np.all(np.isin(classes, [0, 1])):
            raise ValueError("This implementation supports only binary classification with labels 1 and 2.")

        # Indices for each class
        ix0 = np.where(Y_train == 0)[0]
        ix1 = np.where(Y_train == 1)[0]

        # Class priors
        Pw0 = len(ix0) / (len(ix0) + len(ix1))
        Pw1 = 1 - Pw0

        # Fit one-component Gaussian Mixture Models
        clf0 = mixture.GaussianMixture(n_components=1)
        clf1 = mixture.GaussianMixture(n_components=1)

        mod0 = clf0.fit(X_train.iloc[ix0])
        mod1 = clf1.fit(X_train.iloc[ix1])

        self.classifier = {
            'mean0': mod0.means_.flatten(),
            'mean1': mod1.means_.flatten(),
            'cov0': mod0.covariances_[0],
            'cov1': mod1.covariances_[0],
            'Pw0': Pw0,
            'Pw1': Pw1
        }

    
    def objective_function(self, X_test):
        """
        Compute the posterior probabilities for each class.
        Returns an (n_samples, 2) array where columns are P(w1|x), P(w2|x).
        """
       
        Pw0X = multivariate_normal.pdf(X_test, mean=self.classifier['mean0'], cov=self.classifier['cov0']) * self.classifier['Pw0']
        Pw1X = multivariate_normal.pdf(X_test, mean=self.classifier['mean1'], cov=self.classifier['cov1']) * self.classifier['Pw1']
        
        epsilon = 1e-12  # Prevent division by zero
        total = Pw0X + Pw1X + epsilon

        # Normalize to get posterior probabilities
        P0 = Pw0X / total
        P1 = Pw1X / total

        return P1  # shape (n_samples, 2)
    
        
    def predict_proba(self, X_test):
        """
        Compute the posterior probabilities for each class.
        Returns an (n_samples, 2) array where columns are P(w1|x), P(w2|x).
        """
       
        Pw0X = multivariate_normal.pdf(X_test, mean=self.classifier['mean0'], cov=self.classifier['cov0']) * self.classifier['Pw0']
        Pw1X = multivariate_normal.pdf(X_test, mean=self.classifier['mean1'], cov=self.classifier['cov1']) * self.classifier['Pw1']
        
        epsilon = 1e-12  # Prevent division by zero
        total = Pw0X + Pw1X + epsilon

        # Normalize to get posterior probabilities
        P0 = Pw0X / total
        P1 = Pw1X / total

        return np.vstack([P0, P1]).T  # shape (n_samples, 2)
    

    
   

    def predict(self, X_test):
        """
        Predict class labels for input data.
        Returns 1 or 2 for each input sample.
        """
        probs = self.predict_proba(X_test)
        return np.where(probs[:, 0] > probs[:, 1], 0, 1)
