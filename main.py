import pandas as pd
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pre_process_funcs import normalize_features, drop_categorical_variables, drop_binary_variables
from feature_analysis_funcs import kruskal_wallis, generate_cov_matrix, remove_worst_features, analyse_pca, analyse_lda, rank_features_by_auc
from crossvalidation import cross_validate_classifier, calculate_specificity
from roc_objective_functions import euclidean_objective_function, mahalanobis_objective_function, fisher_objective_function, knn_objective_function

from euclidean_MDC import Euclidean_MDC
from mahalanobis_MDC import Mahalanobis_MDC
from LDA_fisher_MDC import LDA_Fisher_MDC
from KNN_classifier import KnnClassifier
from SVM_classifier import SvmClassifier

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
#6. rank de features com kruskal wallis
kw_ranking = kruskal_wallis(X, Y)

#____________________________________________________________________________________

#6.1. rank de features com ROC AUC
auc_ranking = rank_features_by_auc(X,Y)


#____________________________________________________________________________________

#7. remover as 30% piores features com base no teste de kruskal wallis
X = remove_worst_features(X, kw_ranking, percentage=0.3)
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

classifiers = [Euclidean_MDC(), Mahalanobis_MDC(), LDA_Fisher_MDC(), KnnClassifier()]

for classifier in classifiers: 
    cross_validate_classifier(X_pca, Y, classifier, 5)



#_____________________________________________________________________________________
#11. find optimal k clusters for knn classifier
#knn_analysis(X_pca, Y, np.arange(1,12,2), 3)
