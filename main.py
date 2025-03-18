import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import numpy as np
import plotly.express as px

def drop_categorical_variables(X): 
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    print(f"Filtering categoric variables: {categorical_columns[1:]}")
    return X.drop(columns=categorical_columns)

def drop_binary_variables(X):
    binary_columns = [col for col in X.columns if X[col].nunique() == 2 and X[col].dropna().isin([0, 1]).all()]
    print(f"Filtering binary columns: {binary_columns}")
    return X.drop(columns=binary_columns)

def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(features)  # Normalize the values
    normalized_df = pd.DataFrame(normalized_array, columns=features.columns)  # Convert back to DataFrame
    print("Normalizing dataset...")
    return normalized_df

def kruskal_wallis(X, Y):

    fnames=X.columns[1:]

    X=X.to_numpy()[:,1:]

    ixPhishing = np.where(Y==1)
    ixNotPhishing = np.where(Y==0)
   
    Hs={}

    for i in range(np.shape(X)[1]):
        st=stats.kruskal(X[ixPhishing,i].flatten(), X[ixNotPhishing,i].flatten())
        Hs[fnames[i]]=st

    
    Hs = sorted(Hs.items(), key=lambda x: x[1],reverse=True)  

    print("Ranked features using Kruskal Wallis")

    for f in Hs:
        print(f[0]+"-->"+str(f[1]))

    return Hs


def generate_cov_matrix(X, show_img=True):

    feature_names = X.columns.tolist()

    X = X.to_numpy()
    
    # Compute correlation matrix for all features
    corrMat = np.corrcoef(X.T)  # Transpose to ensure correct feature-wise correlation

    # Get feature names
   

    # Plot the correlation matrix using Plotly
    if show_img:
        fig = px.imshow(corrMat, 
                        text_auto=True,
                        labels=dict(x="Features", y="Features", color="Correlation"),
                        x=feature_names,
                        y=feature_names,
                        width=1000, height=1000,
                        color_continuous_scale=px.colors.sequential.Viridis)  # You can change the color scheme
        
        fig.show()

    corrMat = pd.DataFrame(corrMat, index=feature_names, columns=feature_names)

    # Find feature pairs with correlation above threshold
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):  # Only upper triangle
            if abs(corrMat.iloc[i, j]) > 0.8:
                high_corr_pairs.append((feature_names[i], feature_names[j], corrMat.iloc[i, j]))

    # Sort by absolute correlation value (descending order)
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    for pair in high_corr_pairs:
        print(f"Features: {pair[0]} & {pair[1]} --> Correlation: {pair[2]:.2f}")

    return corrMat, high_corr_pairs


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

#7. now let's remove the 20% worse features according to kruskal_wallis



#6. PCA e LDA sobre as selected features para criar novas features