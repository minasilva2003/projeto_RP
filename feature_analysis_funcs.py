from scipy import stats
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd

#function to perform kruskal_wallis test for feature selection
def kruskal_wallis(X, Y):

    fnames=X.columns[1:]

    X=X.to_numpy()[:,1:]

    ixLegitimate = np.where(Y==1)
    ixNotLegitimate = np.where(Y==0)
   
    Hs={}

    for i in range(np.shape(X)[1]):
        st=stats.kruskal(X[ixLegitimate,i].flatten(), X[ixNotLegitimate,i].flatten())
        Hs[fnames[i]]=st

    
    Hs = sorted(Hs.items(), key=lambda x: x[1],reverse=True)  

    print("Ranked features using Kruskal Wallis")

    for f in Hs:
        print(f[0]+"-->"+str(f[1]))

    return Hs



#function to generate covariance matrix and return highly correlated feature pairs
def generate_cov_matrix(X, show_img=True):

    feature_names = X.columns.tolist()

    X = X.to_numpy()
    
    # Compute correlation matrix for all features
    corrMat = np.corrcoef(X.T)  # Transpose to ensure correct feature-wise correlation

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


#function to remove worst features based on kruskal_wallis test
def remove_worst_features(X, Hs, percentage=0.2):
    """
    Removes the worst `percentage` fraction of features based on Kruskal-Wallis test scores.
    
    Parameters:
        X (pd.DataFrame): Feature dataset.
        Hs (list of tuples): Ranked list of features from Kruskal-Wallis test, sorted from best to worst.
        percentage (float): Fraction of worst features to remove (default is 20%).
    
    Returns:
        pd.DataFrame: Dataset with the worst features removed.
    """
    
    num_features_to_remove = int(len(Hs) * percentage)
    worst_features = [Hs[-(i+1)][0] for i in range(num_features_to_remove)]  # Get the worst features
    
    print(f"Removing {num_features_to_remove} worst features: {worst_features}")
    
    return X.drop(columns=worst_features)


#function to perform PCA and check which components to keep based on kaiser test
def analyse_pca(X, show_img=True):
    pca = PCA()
    pca.fit(X)

    #PCA eigenvalues/Explained variance
    print("PCA eigenvalues/Explained variance")
    print(pca.explained_variance_)
    print("Sum of eigenvalues="+str(np.sum(pca.explained_variance_)))
    
    #PCA eigenvectors/Principal components
    print("PCA eigenvectors/Principal components")
    W=pca.components_.T
    print(W)

    print("The main PC contributes to "+str(pca.explained_variance_[0]**2/(pca.explained_variance_[0]**2+pca.explained_variance_[1]**2)*100)+"% of the variance.")
    
    #Kaiser test (quantos componentes têm eigenvalue acima de 1)
    optimal_components =sum(pca.explained_variance_>1)
    print("Optimal number of components according to Kaiser: "+str(optimal_components))

    #visualizar kaiser test
    if show_img:
        fig = px.scatter(x=np.arange(1,len(pca.explained_variance_)+1,1), y=pca.explained_variance_,
                    labels=dict(x="PC",y="Explained Variance"))
        fig.add_hline(y=1,line_width=3, line_dash="dash", line_color="red")
        fig.update_traces(marker_size=10)
        fig.show()

    print("Variance (%) retained accourding to Kaiser: "+str(pca.explained_variance_[0]**2/(np.sum(pca.explained_variance_**2))*100))
    print("Variance (%) retained accourding to Scree: "+str(np.sum(pca.explained_variance_[0:6]**2)/(np.sum(pca.explained_variance_**2))*100))

    # Apply PCA with the optimal number of components
    pca_optimal = PCA(n_components=optimal_components)
    X_pca = pca_optimal.fit_transform(X)

    # Return transformed dataset as DataFrame
    return pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(optimal_components)])
    


def analyse_lda(X, Y):
    """
    Applies Linear Discriminant Analysis (LDA) for feature transformation.

    Args:
        X (pd.DataFrame): The feature matrix.
        Y (pd.Series): The target variable (class labels).

    Returns:
        pd.DataFrame: The transformed feature matrix after applying LDA.
    """
    # Initialise the LDA model
    lda = LDA()

    # Fit the LDA model to the data and transform it
    X_transformed = lda.fit_transform(X, Y)

    # Convert the transformed array back to a Pandas DataFrame
    X_transformed_df = pd.DataFrame(X_transformed, index=X.index)
    X_transformed_df.columns = [f'LDA_Component_{i+1}' for i in range(X_transformed.shape[1])]

    return X_transformed_df


