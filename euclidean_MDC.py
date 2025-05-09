import numpy as np
from scipy.spatial.distance import euclidean


class Euclidean_MDC:
    """
    An euclidean minimum distance classifier
    """

    def __init__(self, data_info):
        self.class_means = {}
        self.classifier_label = "Euclidean_MDC"
        self.data_info = data_info
        
    def train(self, X_train, Y_train):
        """
        Train the classifier
        """
        
        # Ensure inputs are numpy arrays for easier calculations
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)

        # Calculate the mean of features for each class in the training data
        class_labels = np.unique(Y_train)

        for label in class_labels:
            self.class_means[label] = np.mean(X_train[Y_train == label], axis=0)

    def predict(self, X_test):
        """
        Predict labels for a set of data
        """
        
        X_test = np.asarray(X_test)

        # Predict the class for each test data point based on Euclidean distance
        predictions = []
        for test_point in X_test:
            distances = {}
            for label, mean in self.class_means.items():
                distances[label] = euclidean(test_point, mean)

            # The predicted class is the one with the minimum Euclidean distance
            predicted_label = min(distances, key=distances.get)
            predictions.append(predicted_label)

        return np.array(predictions)
    

    def objective_function(self, X_test):
        """
        Returns continuous values that reflect confidence level of each point 
        in dataset belonging to positive class
        """
        return -np.linalg.norm(X_test - self.class_means[1], axis=1)  # more negative = less confident
