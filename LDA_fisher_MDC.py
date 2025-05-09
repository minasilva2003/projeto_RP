import numpy as np
from scipy.spatial.distance import euclidean


class LDA_Fisher_MDC:
    """
    An LDA Fisher classifier
    """

    def __init__(self, data_info):
        self.m0 = None
        self.m1 = None
        self.projection_vector = None
        self.m0_proj = None
        self.m1_proj = None
        self.classifier_label = "LDA_Fisher_MDC"
        self.data_info = data_info

    def train(self, X_train, Y_train):
        
        """
        Train the classifier
        """
        
        # Ensure numpy arrays
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
      
        # Compute class means and scatter matrices
        class0 = X_train[Y_train == 0]
        class1 = X_train[Y_train == 1]

        self.m0 = np.mean(class0, axis=0)
        self.m1 = np.mean(class1, axis=0)

        # Compute within-class scatter matrix Sw
        S0 = np.cov(class0, rowvar=False)
        S1 = np.cov(class1, rowvar=False)
        Sw = S0 + S1
    
        # Compute LDA projection vector
        self.projection_vector = np.linalg.inv(Sw).dot(self.m1 - self.m0)
        self.m0_proj = np.dot(self.m0, self.projection_vector)
        self.m1_proj = np.dot(self.m1, self.projection_vector)
        


    def predict(self, X_test):
        """
        Predict labels for a set of data
        """
        X_test = np.asarray(X_test)
        X_test_proj = X_test @ self.projection_vector

        # Classify based on minimum Euclidean distance in 1D
        predictions = [0 if abs(x - self.m0_proj) < abs(x - self.m1_proj) else 1 for x in X_test_proj]

        return np.array(predictions)
        

    def objective_function(self, X_test):
        """
        Returns continuous values that reflect confidence level of each point 
        in dataset belonging to positive class
        """
        X_test = np.asarray(X_test)

        scores = X_test @ self.projection_vector  # projection values
        return scores

