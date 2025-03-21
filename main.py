import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from scipy.linalg import svd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pre_process_funcs import normalize_features, drop_categorical_variables, drop_binary_variables
from feature_analysis_funcs import kruskal_wallis, generate_cov_matrix, remove_worst_features, analyse_pca, analyse_lda
from min_dist_classifiers import min_euclidean_distance_classifier, min_mahalanobis_distance_classifier, calculate_sensitivity, calculate_specificity
#________________________________________________________________________________

#0. ler dataset
url_dataset = pd.read_csv("dataset.csv") 
print(url_dataset.shape)
X = url_dataset.drop(columns=["label"])
Y = url_dataset.iloc[:,-1]

#_________________________________________________________________________________

#1. remover categóricas e binárias
print(X.shape)
X = drop_categorical_variables(X)
print(X.shape)
X = drop_binary_variables(X)
print(X.shape)

#_________________________________________________________________________________

#2. limpar dataset
#quando se faz url_dataset.variables ele supostamente diz que nenhuma coluna tem missing data portanto não sei se isto é necessário

#__________________________________________________________________________________

#3. normalizar/scaling
X = normalize_features(X)

#___________________________________________________________________________________

#4. covariance matrix
covm, high_corr_pairs = generate_cov_matrix(X, show_img=False)

#____________________________________________________________________________________

#5. using the information from the covariance matrix, let's remove the following features:
X = X.drop(columns=["DomainTitleMatchScore", "NoOfLettersInURL", "NoOfDegitsInURL"])
print(X.shape)


#____________________________________________________________________________________
#6. kriskow walix para feature selection
Hs = kruskal_wallis(X, Y)

#____________________________________________________________________________________

#7. now let's remove the 30% worse features according to kruskal_wallis
X = remove_worst_features(X, Hs, percentage=0.3)
print(X.shape)


#____________________________________________________________________________________

#____________________________________________________________________________________
#9. PCA e LDA sobre as selected features para criar novas features
X_pca = analyse_pca(X, show_img=False)
print(X_pca.shape)

#X_lda = analyse_lda(X, Y)
#print(X_lda.shape)


#____________________________________________________________________________________
#10. min distance classifiers
# Split the data into full training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

n_folds = 5
euclidean_accuracy_scores = []
euclidean_sensitivity_scores = []
mahalanobis_accuracy_scores = []
mahalanobis_sensitivity_scores = []
fold_size = len(X_train_full) // n_folds
indices = np.arange(len(X_train_full))

print("\n*** Cross-validation for Euclidean Distance Classifier ***")
for i in range(n_folds):
    # Create validation set and training set for this fold
    start = i * fold_size
    end = (i + 1) * fold_size
    if i == n_folds - 1:
        val_indices = indices[start:]
        train_indices = np.concatenate((indices[:start]))
    else:
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

    X_train_fold, X_val_fold = X_train_full[train_indices], X_train_full[val_indices]
    y_train_fold, y_val_fold = y_train_full[train_indices], y_val_full[val_indices]

    # Train and evaluate Euclidean classifier
    y_pred_euclidean = min_euclidean_distance_classifier(X_train_fold, y_train_fold, X_val_fold)
    accuracy_euclidean = accuracy_score(y_val_fold, y_pred_euclidean)
    sensitivity_euclidean = calculate_sensitivity(y_val_fold, y_pred_euclidean)
    euclidean_accuracy_scores.append(accuracy_euclidean)
    euclidean_sensitivity_scores.append(sensitivity_euclidean)
    print(f"Fold {i+1} Euclidean Accuracy: {accuracy_euclidean:.2f}, Sensitivity: {sensitivity_euclidean:.2f}")

    # Train and evaluate Mahalanobis classifier
    y_pred_mahalanobis = min_mahalanobis_distance_classifier(X_train_fold, y_train_fold, X_val_fold)
    accuracy_mahalanobis = accuracy_score(y_val_fold, y_pred_mahalanobis)
    sensitivity_mahalanobis = calculate_sensitivity(y_val_fold, y_pred_mahalanobis)
    mahalanobis_accuracy_scores.append(accuracy_mahalanobis)
    mahalanobis_sensitivity_scores.append(sensitivity_mahalanobis)
    print(f"Fold {i+1} Mahalanobis Accuracy: {accuracy_mahalanobis:.2f}, Sensitivity: {sensitivity_mahalanobis:.2f}")

print("\n*** Summary of Euclidean Distance Classifier Cross-validation ***")
print(f"Cross-validation Accuracy Scores: {euclidean_accuracy_scores}")
print(f"Mean Cross-validation Accuracy: {np.mean(euclidean_accuracy_scores):.2f}")
print(f"Standard Deviation of Cross-validation Accuracy: {np.std(euclidean_accuracy_scores):.2f}")
print(f"Cross-validation Sensitivity Scores: {euclidean_sensitivity_scores}")
print(f"Mean Cross-validation Sensitivity: {np.mean(euclidean_sensitivity_scores):.2f}")
print(f"Standard Deviation of Cross-validation Sensitivity: {np.std(euclidean_sensitivity_scores):.2f}")

print("\n*** Summary of Mahalanobis Distance Classifier Cross-validation ***")
print(f"Cross-validation Accuracy Scores: {mahalanobis_accuracy_scores}")
print(f"Mean Cross-validation Accuracy: {np.mean(mahalanobis_accuracy_scores):.2f}")
print(f"Standard Deviation of Cross-validation Accuracy: {np.std(mahalanobis_accuracy_scores):.2f}")
print(f"Cross-validation Sensitivity Scores: {mahalanobis_sensitivity_scores}")
print(f"Mean Cross-validation Sensitivity: {np.mean(mahalanobis_sensitivity_scores):.2f}")
print(f"Standard Deviation of Cross-validation Sensitivity: {np.std(mahalanobis_sensitivity_scores):.2f}")
