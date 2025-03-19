import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
from scipy.linalg import svd
from pre_process_funcs import normalize_features, drop_categorical_variables, drop_binary_variables
from feature_analysis_funcs import kruskal_wallis, generate_cov_matrix, remove_worst_features, analyse_pca
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
X = analyse_pca(X)
print(X.shape)