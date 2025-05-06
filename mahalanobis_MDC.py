import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv

class Mahalanobis_MDC:
    """
    A mahalanobis distance minimum distance classifier
    """

    def __init__(self):
        self.class_means = {}
        self.class_labels = None
        self.class_covariances = {}
        self.class_inv_covariances = {}
        self.classifier_label = "Mahalanobis MDC"

    def train(self, X_train, Y_train):
        """
        Train the classifier
        """
        
        # Ensure inputs are numpy arrays for easier calculations
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
       
        self.class_labels = np.unique(Y_train)

        for label in self.class_labels:

            # 1. Calculate the mean vector 
            X_class = X_train[Y_train == label]
            self.class_means[label] = np.mean(X_class, axis=0)
            
            # 2. Calculate the covariance matrix for each class
            cov = np.cov(X_class, rowvar=False)
            self.class_covariances[label] = np.atleast_2d(cov)  # Ensures 2D structure

            # 3. Calculate the inverse of the covariance matrix; handle potential singular matrices
            try:
                self.class_inv_covariances[label] = inv(self.class_covariances[label])
            except Exception as e:
                print(f"Warning: Covariance matrix for class {label} is singular. Using pseudo-inverse.")
                self.class_inv_covariances[label] = np.linalg.pinv(self.class_covariances[label])


    def predict(self, X_test):
        """
        Predict labels for a set of data
        """

        # 2. Predict the class for each test data point based on Mahalanobis distance
        X_test = np.asarray(X_test)
        predictions = []

        for test_point in X_test:
            
            distances = {}

            for label in self.class_labels:
                # Calculate the Mahalanobis distance to the mean of the current class
                mean_vector = self.class_means[label]
                inv_covariance = self.class_inv_covariances[label]
                distances[label] = mahalanobis(np.asarray(test_point), np.asarray(mean_vector), inv_covariance)

            # The predicted class is the one with the minimum Mahalanobis distance
            predicted_label = min(distances, key=distances.get)
            predictions.append(predicted_label)

        return np.array(predictions)


    def objective_function(self, X_test):
        """
        Returns continuous values that reflect confidence level of each point 
        in dataset belonging to positive class
        """
        
           # Score = -mahalanobis distance to phishing mean
        scores = [-mahalanobis(np.asarray(x), np.asarray(self.class_means[1]), self.class_inv_covariances[1]) for x in np.asarray(X_test)]
        return np.array(scores)
