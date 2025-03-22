import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pre_process_funcs import normalize_features, drop_categorical_variables, drop_binary_variables
from feature_analysis_funcs import kruskal_wallis, generate_cov_matrix, remove_worst_features, analyse_pca, analyse_lda
from min_dist_classifiers import min_euclidean_distance_classifier, min_mahalanobis_distance_classifier, calculate_sensitivity, calculate_specificity, fisher_lda_classifier
#________________________________________________________________________________

#0. ler dataset
url_dataset = pd.read_csv("dataset.csv") 
print(url_dataset.shape)
X = url_dataset.drop(columns=["label"])
Y = url_dataset.iloc[:,-1]

#_________________________________________________________________________________

#1. remover features categóricas e binárias
print(X.shape)
X = drop_categorical_variables(X)
print(X.shape)
X = drop_binary_variables(X)
print(X.shape)

#_________________________________________________________________________________

#2. limpar dataset
#informação disponibilizada sobre o dataset garante que não há valores em falta

#__________________________________________________________________________________

#3. normalizar/scaling
X = normalize_features(X)

#___________________________________________________________________________________

#4. covariance matrix
covm, high_corr_pairs = generate_cov_matrix(X, show_img=False)

#____________________________________________________________________________________

#5. com a informação da covariance matrix, removemos as seguintes features
X = X.drop(columns=["DomainTitleMatchScore", "NoOfLettersInURL", "NoOfDegitsInURL"])
print(X.shape)


#____________________________________________________________________________________
#6. kruskal wallis para seleção de features
Hs = kruskal_wallis(X, Y)

#____________________________________________________________________________________

#7. remover as 30% piores features com base no teste de kruskal wallis
X = remove_worst_features(X, Hs, percentage=0.3)
print(X.shape)


#____________________________________________________________________________________

#____________________________________________________________________________________
#9. PCA e LDA sobre as selected features para criar novas features
X_pca = analyse_pca(X, show_img=False)
print(X_pca.shape)

X_lda = analyse_lda(X, Y)
print(X_lda.shape)


#____________________________________________________________________________________
#10. min distance classifiers com distância euclidiana e de mahalanobis
# Podemos usar o dataset com pca ou lda
X_train_full, X_test, y_train_full, y_test = train_test_split(X_lda, Y, test_size=0.3, random_state=42)

n_folds = 5
euclidean_accuracy_scores = []
euclidean_specificity_scores = []
mahalanobis_accuracy_scores = []
mahalanobis_specificity_scores = []
fold_size = len(X_train_full) // n_folds
indices = np.arange(len(X_train_full))

print("\n*** Cross-validation for Euclidean Distance Classifier ***")
for i in range(n_folds):
    # Criar folds
    start = i * fold_size
    end = (i + 1) * fold_size
    if i == n_folds - 1:
        val_indices = indices[start:]
        train_indices = indices[:start]
    else:
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

    X_train_fold, X_val_fold = X_train_full.iloc[train_indices], X_train_full.iloc[val_indices]
    y_train_fold, y_val_fold = y_train_full.iloc[train_indices], y_train_full.iloc[val_indices]

    # Treinar e avaliar classificador euclidiano
    y_pred_euclidean = min_euclidean_distance_classifier(X_train_fold, y_train_fold, X_val_fold)
    accuracy_euclidean = accuracy_score(y_val_fold, y_pred_euclidean)
    specificity_euclidean = calculate_specificity(y_val_fold, y_pred_euclidean)
    euclidean_accuracy_scores.append(accuracy_euclidean)
    euclidean_specificity_scores.append(specificity_euclidean)
   
    # Treinar e avaliar classificador mahalanobis
    y_pred_mahalanobis = min_mahalanobis_distance_classifier(X_train_fold, y_train_fold, X_val_fold)
    accuracy_mahalanobis = accuracy_score(y_val_fold, y_pred_mahalanobis)
    specificity_mahalanobis = calculate_specificity(y_val_fold, y_pred_mahalanobis)
    mahalanobis_accuracy_scores.append(accuracy_mahalanobis)
    mahalanobis_specificity_scores.append(specificity_mahalanobis)
    
print("\n*** Summary of Euclidean Distance Classifier Cross-validation ***")
print(f"Mean Cross-validation Accuracy: {np.mean(euclidean_accuracy_scores):.3f}")
print(f"Standard Deviation of Cross-validation Accuracy: {np.std(euclidean_accuracy_scores):.3f}")
print(f"Mean Cross-validation Specificity: {np.mean(euclidean_specificity_scores):.3f}")
print(f"Standard Deviation of Cross-validation Specificity: {np.std(euclidean_specificity_scores):.3f}")

print("\n*** Summary of Mahalanobis Distance Classifier Cross-validation ***")
print(f"Mean Cross-validation Accuracy: {np.mean(mahalanobis_accuracy_scores):.3f}")
print(f"Standard Deviation of Cross-validation Accuracy: {np.std(mahalanobis_accuracy_scores):.3f}")
print(f"Mean Cross-validation Specificity: {np.mean(mahalanobis_specificity_scores):.3f}")
print(f"Standard Deviation of Cross-validation Specificity: {np.std(mahalanobis_specificity_scores):.3f}")


#____________________________________________________________________________________
#11. Fisher LDA
X_train_full, X_test, y_train_full, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

n_folds = 5
fisher_accuracy_scores = []
fisher_specificity_scores = []
fold_size = len(X_train_full) // n_folds
indices = np.arange(len(X_train_full))

print("\n*** Cross-validation for FISHER LDA Classifier ***")
for i in range(n_folds):
    # Criar fold
    start = i * fold_size
    end = (i + 1) * fold_size
    if i == n_folds - 1:
        val_indices = indices[start:]
        train_indices = indices[:start]
    else:
        val_indices = indices[start:end]
        train_indices = np.concatenate((indices[:start], indices[end:]))

    X_train_fold, X_val_fold = X_train_full.iloc[train_indices], X_train_full.iloc[val_indices]
    y_train_fold, y_val_fold = y_train_full.iloc[train_indices], y_train_full.iloc[val_indices]

    # Treinar e avaliar classificador Fisher
    y_pred_fisher = fisher_lda_classifier(X_train_fold, y_train_fold, X_val_fold)
    accuracy_fisher = accuracy_score(y_val_fold, y_pred_fisher)
    specificity_fisher = calculate_specificity(y_val_fold, y_pred_fisher)
    fisher_accuracy_scores.append(accuracy_fisher)
    fisher_specificity_scores.append(specificity_fisher)
    
print("\n*** Summary of Fisher Classifier Cross-validation ***")
print(f"Mean Cross-validation Accuracy: {np.mean(fisher_accuracy_scores):.3f}")
print(f"Standard Deviation of Cross-validation Accuracy: {np.std(fisher_accuracy_scores):.3f}")
print(f"Mean Cross-validation Specificity: {np.mean(fisher_specificity_scores):.3f}")
print(f"Standard Deviation of Cross-validation Specificity: {np.std(fisher_specificity_scores):.3f}")

