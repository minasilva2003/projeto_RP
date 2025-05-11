import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import time

log_file_path = "main.log"

def print_log(message):
    """
    Appends a message to a .log file.

    Parameters:
        log_file_path (str): Path to the log file.
        message (str): The message to append.
    """

    print(message)

    with open(log_file_path, 'a') as log_file:
        log_file.write(message + '\n')

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



def cross_validate_classifier(X, y, classifier, run, n_folds=5, view=True):
    """
    Perform cross-validation with K folds for a given classifier.
    
    Parameters:
        X (pd.DataFrame): Feature set.
        y (pd.Series): Corresponding labels (0 or 1).
        classifier (class): The classifier class to be used (must have train and predict methods).
        n_folds (int): Number of folds for cross-validation. Default = 5.
        view (bool): Whether to plot the ROC curve (default is True).
    
    Returns:
        tuple: (mean accuracy, mean specificity, mean F1-score, mean sensitivity, mean AUC score)
    """
  
    # Create shuffled indices for cross-validation
    indices = np.arange(len(X))
    fold_size = len(X) // n_folds
    
    np.random.shuffle(indices)

    # Initialize lists to store results for each fold
    accuracy_scores = []
    specificity_scores = []
    f1_scores = []
    sensitivity_scores = []
    auc_scores = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    print(f"\n*** Cross-validation #{run} for {classifier.classifier_label} ***")
  
    # Initialize the scaler
    scaler = MinMaxScaler()

    for i in range(n_folds):
        # Define the indices for the current validation fold
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(X)  # last fold might have different size

        val_indices = indices[start:end]  # validation data
        train_indices = np.concatenate([indices[:start], indices[end:]])  # training data

        # Split the data into training and validation based on indices
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

        # Fit the scaler on the training data and transform both train and validation sets
        X_train = pd.DataFrame(scaler.fit_transform(X_train))
        X_val = pd.DataFrame(scaler.transform(X_val))

        # Train the classifier on the scaled training data
        classifier.train(X_train, y_train)
        
        # Predict the class labels for the validation set
        y_pred = classifier.predict(X_val)

        # Calculate performance metrics
        acc = accuracy_score(y_val, y_pred)
        spec = calculate_specificity(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        sensitivity = recall_score(y_val, y_pred)  # Sensitivity is the same as recall

        # Store the results
        accuracy_scores.append(acc)
        specificity_scores.append(spec)
        f1_scores.append(f1)
        sensitivity_scores.append(sensitivity)

        # Calculate the ROC curve if possible
        if classifier.objective_function is not None:
            try:
                y_scores = classifier.objective_function(X_val)
                fpr, tpr, _ = roc_curve(y_val, y_scores)
                roc_auc = auc(fpr, tpr)
                auc_scores.append(roc_auc)
            
                # Interpolation for average ROC curve
                tpr_interp = np.interp(mean_fpr, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)

            except Exception as e:
                print("error computing auc")
               
    # Show aggregated results (mean and std)
    print_log(f"Mean Accuracy for {classifier.classifier_label}: {np.mean(accuracy_scores):.3f} ± {np.std(accuracy_scores):.3f}")
    print_log(f"Mean Specificity for {classifier.classifier_label}: {np.mean(specificity_scores):.3f} ± {np.std(specificity_scores):.3f}")
    print_log(f"Mean F1-score for {classifier.classifier_label}: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
    print_log(f"Mean Sensitivity (Recall) for {classifier.classifier_label}: {np.mean(sensitivity_scores):.3f} ± {np.std(sensitivity_scores):.3f}")
    
    # Plot the mean ROC curve (if AUC scores are available and objective function exists)
    if len(auc_scores) > 0 and classifier.objective_function is not None:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        print_log(f"Mean AUC for {classifier.classifier_label}: {mean_auc}")
        print_log("\n\n#########################################################################\n\n")

        plt.figure(figsize=(8, 6))
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=f'{classifier.classifier_label} (AUC = {mean_auc:.2f})', lw=2)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Mean ROC Curve - {classifier.classifier_label} - {classifier.data_info}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()

        #save figure
        #image_directory = f"auc_curves/{classifier.classifier_label}/{classifier.data_info}/"
        #os.makedirs(image_directory, exist_ok=True)
        #plt.savefig(image_directory + f'curve_{classifier.classifier_label}_{classifier.data_info}_{run}.png')  # You can specify the path and format, e.g., .jpg, .png, etc.
        if view:
            plt.show()

        return [np.mean(accuracy_scores), np.mean(specificity_scores), np.mean(f1_scores), 
                np.mean(sensitivity_scores), mean_auc]
        
    print_log("\n\n#########################################################################\n\n")

    return [np.mean(accuracy_scores), np.mean(specificity_scores), np.mean(f1_scores), 
            np.mean(sensitivity_scores), None]