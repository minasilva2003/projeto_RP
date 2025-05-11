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
from BayesianClass import BayesianGaussianClassifier

import csv

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


#________________________________________________________________________________

#0. read dataset
url_dataset = pd.read_csv("dataset.csv") 
X = url_dataset.drop(columns=["label"])
Y = url_dataset.iloc[:,-1]

print_log("\n\n#########################################################################\n\n")
print_log(f"Shape of original X: {X.shape}")
print_log("\n\n#########################################################################\n\n")

#_________________________________________________________________________________

#1. remove categorical and binary features
X = drop_categorical_variables(X)

print_log(f"Shape of X after dropping categorical variables: {X.shape}")
print_log("\n\n#########################################################################\n\n")

X = drop_binary_variables(X)

print_log(f"Shape of X after dropping binary variables: {X.shape}")
print_log("\n\n#########################################################################\n\n")

#_________________________________________________________________________________

#2. Cleaning the dataset
#Avaliable information about the dataset guarantees that there are no missing values

#__________________________________________________________________________________

#3. normalize/scaling
#X = normalize_features(X)

#___________________________________________________________________________________

#4. covariance matrix
covm, high_corr_pairs = generate_cov_matrix(X, show_img=False)

#____________________________________________________________________________________

#5. with the information from the covariance matrix, we remove the following features
X = X.drop(columns=["DomainTitleMatchScore", "NoOfLettersInURL", "NoOfDegitsInURL"])
print_log(f"Shape of X after dropping highly correlated variables: {X.shape}")
print_log("\n\n#########################################################################\n\n")

#____________________________________________________________________________________
#6. Feature ranking with kruskal wallis

print_log("Performing Kruskal Wallis Ranking...")
kw_ranking = kruskal_wallis(X, Y)

#____________________________________________________________________________________

#6.1. Feature ranking with ROC AUC
print_log("\n\n#########################################################################\n\n")
print_log("Performing AUC ranking...")
auc_ranking = rank_features_by_auc(X,Y)


#____________________________________________________________________________________

#7. remove the 20% worst features based on kruskal wallis and roc auc
kw_X = remove_worst_features(X, kw_ranking, percentage=0.2)
auc_X = remove_worst_features(X, auc_ranking, percentage=0.2)

print_log("\n\n#########################################################################\n\n")
print_log(f"Shape for dataset after KW ranking selection: {kw_X.shape}")
print_log(f"Shape for dataset after AUC ranking selection: {auc_X.shape}")
print_log("\n\n#########################################################################\n\n")

feature_selection_Xs = {"KW": kw_X, "AUC": auc_X}

#____________________________________________________________________________________

rows = [["Selection Ranking", "Processing", "Classifier", "Run", "Accuracy", "Specificity", "F1-Score", "Sensitivity", "Auc"]]

for feature_selection_type, fs_X in feature_selection_Xs.items():
    
    print_log("\n\n#########################################################################\n\n")
    
    print_log(f"Performing PCA for {feature_selection_type}...")
    
    #9. PCA and LDA on the selected features to create new features
    X_pca = analyse_pca(fs_X, feature_selection_type, show_img=False)
    
    print_log(f"Shape of dataset after PCA for {feature_selection_type} is: {X_pca.shape}")

    print_log("\n\n#########################################################################\n\n")
    
    print_log(f"Performing LDA for {feature_selection_type}...")
    
    X_lda = analyse_lda(fs_X, Y)
    
    print_log(f"Shape of dataset after LDA is {feature_selection_type}: {X_lda.shape}")

    print_log("\n\n#########################################################################\n\n")

    data_processing_Xs = {"Natural": fs_X, "PCA": X_pca, "LDA": X_lda}

    #10. Loop of experiments with feature selection + data processing + classifier
    for data_processing_type, processed_X in data_processing_Xs.items():

        data_info_string = feature_selection_type + "_" + data_processing_type

        print_log(f"TESTING CLASSIFIERS FOR DATASET {data_info_string}...")

        #____________________________________________________________________________________
        
        #11. verifying the best K for the classifier KNN
      
        print_log("TESTING KNN...")
        knn_classifier = KnnClassifier(data_info = data_info_string)
        knn_training_results = knn_classifier.knn_analysis(processed_X, Y, n_runs=3)

        # Save results
        df = pd.DataFrame(zip(*knn_training_results))
        df.columns = ['K', 'Average Error', 'STD Error']
        df.to_csv(f"knn_training/err_{data_info_string}.csv", index=False)

        #____________________________________________________________________________________

        #12. verify the best C value for the classifier SVM
        print_log("TESTING SVM...")
        svm_classifier = SvmClassifier(data_info = data_info_string, kernel_function="rbf")
        svm_training_results = svm_classifier.svm_analysis(processed_X, Y, n_runs=3)

        # Save results
        df = pd.DataFrame(zip(*svm_training_results))
        df.columns = ['C', 'Average Error', 'STD Error']
        df.to_csv(f"svm_training/err_{data_info_string}.csv", index=False)

        #13. Selecting which classifiers to use based on the dataset
        if data_processing_type != "LDA":
            classifiers = [Euclidean_MDC(data_info_string), Mahalanobis_MDC(data_info_string), LDA_Fisher_MDC(data_info_string), BayesianGaussianClassifier(data_info_string), knn_classifier, svm_classifier]
        else:
            classifiers = [Euclidean_MDC(data_info_string), Mahalanobis_MDC(data_info_string), BayesianGaussianClassifier(data_info_string), knn_classifier, svm_classifier]

        try:
            #14. for each classifier run 10 times cross-validation and save results
            for classifier in classifiers: 

                for run in range (5):

                    metrics = cross_validate_classifier(processed_X, Y, classifier, run+1, 5, view=False)

                    results = [feature_selection_type, data_processing_type, classifier.classifier_label, run+1] + metrics

                    rows.append(results)
        
        except Exception as e:
            with open('results.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows)  # Write all rows

            


# Writing to a CSV file
with open('results.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)  # Write all rows

