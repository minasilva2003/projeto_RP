import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import pandas as pd


def min_euclidean_distance_classifier(X_train, y_train, X_test):
    """
    Creates and applies a minimum Euclidean distance classifier.

    Args:
        X_train (pd.DataFrame or np.array): Training feature matrix.
        y_train (pd.Series or np.array): Training target variable (class labels).
        X_test (pd.DataFrame or np.array): Test feature matrix.

    Returns:
        np.array: Predicted class labels for the test data.
    """
    # Ensure inputs are numpy arrays for easier calculations
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    # 1. Calculate the mean of features for each class in the training data
    class_labels = np.unique(y_train)
    class_means = {}
    for label in class_labels:
        class_means[label] = np.mean(X_train[y_train == label], axis=0)

    # 2. Predict the class for each test data point based on Euclidean distance
    predictions = []
    for test_point in X_test:
        distances = {}
        for label, mean in class_means.items():
            distances[label] = euclidean(test_point, mean)

        # The predicted class is the one with the minimum Euclidean distance
        predicted_label = min(distances, key=distances.get)
        predictions.append(predicted_label)

    return np.array(predictions)


def min_mahalanobis_distance_classifier(X_train, y_train, X_test):
    """
    Creates and applies a minimum Mahalanobis distance classifier.

    Args:
        X_train (pd.DataFrame or np.array): Training feature matrix.
        y_train (pd.Series or np.array): Training target variable (class labels).
        X_test (pd.DataFrame or np.array): Test feature matrix.

    Returns:
        np.array: Predicted class labels for the test data.
    """
    # Ensure inputs are numpy arrays for easier calculations
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    # 1. Calculate the mean vector and covariance matrix for each class in the training data
    class_labels = np.unique(y_train)
    class_means = {}
    class_covariances = {}
    class_inv_covariances = {}

    for label in class_labels:
        X_class = X_train[y_train == label]
        
        class_means[label] = np.mean(X_class, axis=0)
        # Calculate the covariance matrix for each class
        cov = np.cov(X_class, rowvar=False)
        class_covariances[label] = np.atleast_2d(cov)  # Ensures 2D structure

        # Calculate the inverse of the covariance matrix; handle potential singular matrices
        try:
            class_inv_covariances[label] = inv(class_covariances[label])
        except Exception as e:
            print(f"Warning: Covariance matrix for class {label} is singular. Using pseudo-inverse.")
            class_inv_covariances[label] = np.linalg.pinv(class_covariances[label])

    # 2. Predict the class for each test data point based on Mahalanobis distance
    predictions = []
    for test_point in X_test:
        distances = {}
        for label in class_labels:
            # Calculate the Mahalanobis distance to the mean of the current class
            mean_vector = class_means[label]
            inv_covariance = class_inv_covariances[label]
            distances[label] = mahalanobis(test_point, mean_vector, inv_covariance)

        # The predicted class is the one with the minimum Mahalanobis distance
        predicted_label = min(distances, key=distances.get)
        predictions.append(predicted_label)

    return np.array(predictions)


def calculate_sensitivity(y_true, y_pred, positive_label=1):
    """
    Calculates the sensitivity (True Positive Rate).

    Args:
        y_true (np.array): True class labels.
        y_pred (np.array): Predicted class labels.
        positive_label: The label considered as the positive class.

    Returns:
        float: The sensitivity score.
    """
    true_positives = np.sum((y_true == positive_label) & (y_pred == positive_label))
    false_negatives = np.sum((y_true == positive_label) & (y_pred != positive_label))
    if (true_positives + false_negatives) == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)

def calculate_specificity(y_true, y_pred, negative_label=0):
    """
    Calculates the specificity (True Negative Rate).

    Args:
        y_true (np.array): True class labels.
        y_pred (np.array): Predicted class labels.
        negative_label: The label considered as the negative class.

    Returns:
        float: The specificity score.
    """
    true_negatives = np.sum((y_true == negative_label) & (y_pred == negative_label))
    false_positives = np.sum((y_true == negative_label) & (y_pred != negative_label))
    if (true_negatives + false_positives) == 0:
        return 0.0
    return true_negatives / (true_negatives + false_positives)


def fisher_lda_classifier(X_train, y_train, X_test):
    """
    Fisher's LDA Minimum Distance Classifier.

    Args:
        X_train (np.array): Training feature matrix.
        y_train (np.array): Training labels.
        X_test (np.array): Test feature matrix.

    Returns:
        np.array: Predicted class labels.
    """
    # Ensure numpy arrays
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    # Compute class means
    class_labels = np.unique(y_train)
    m0 = np.mean(X_train[y_train == class_labels[0]], axis=0)
    m1 = np.mean(X_train[y_train == class_labels[1]], axis=0)

    # Compute within-class scatter matrix Sw
    S0 = np.cov(X_train[y_train == class_labels[0]], rowvar=False)
    S1 = np.cov(X_train[y_train == class_labels[1]], rowvar=False)
    Sw = S0 + S1

    # Compute LDA projection vector
    w = np.linalg.inv(Sw).dot(m1 - m0)

    # Project means and test data
    m0_proj = np.dot(m0, w)
    m1_proj = np.dot(m1, w)
    X_test_proj = X_test @ w

    # Classify based on minimum Euclidean distance in 1D
    predictions = [class_labels[0] if abs(x - m0_proj) < abs(x - m1_proj) else class_labels[1] for x in X_test_proj]

    return np.array(predictions)
