from scipy import stats
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import pandas as pd
from sklearn.model_selection import train_test_split
from roc_objective_functions import *
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

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


def kruskal_wallis(X, Y):
    """
    Performs the Kruskal-Wallis test for feature selection by ranking features based on their 
    statistical significance in distinguishing between two classes.

    Parameters:
        X (pd.DataFrame): Feature dataset.
        Y (pd.Series): Target labels (binary classification: 0 or 1).

    Returns:
        list: A ranked list of features (sorted from most to least significant) along with their 
              Kruskal-Wallis test statistics.
    """
    
    fnames = X.columns[1:]  # Exclude the first feature
    X = X.to_numpy()[:,1:]

    ixLegitimate = np.where(Y == 1)
    ixNotLegitimate = np.where(Y == 0)

    Hs = {}

    for i in range(np.shape(X)[1]):
        st = stats.kruskal(X[ixLegitimate, i].flatten(), X[ixNotLegitimate, i].flatten())
        Hs[fnames[i]] = st

    Hs = sorted(Hs.items(), key=lambda x: x[1], reverse=True)  

    res = ""
    for f in Hs:
        res += f[0] + "  --> " + str(f[1]) + "\n"

    print_log("Ranked features using Kruskal-Wallis test:\n" + res)

    return Hs


def rank_features_by_auc(X, y, plot=False):
    """
    Ranks features by their individual ROC-AUC scores.

    Args:
        X (pd.DataFrame): Dataset with only numeric features (after filtering).
        y (pd.Series or np.array): Binary class labels (0 or 1).
        plot (bool): Whether to plot individual ROC curves with AUC annotations.

    Returns:
        list of tuples: [(feature_name, auc), ...] sorted by AUC descending
    """
    feature_names = X.columns
    auc_scores = []

    for feature in feature_names:
        scores = X[feature].to_numpy()

        # ROC and AUC for the current feature
        fpr, tpr, _ = roc_curve(y, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        auc_scores.append((feature, roc_auc))

        # Optional plot
        if plot:
            fig = go.Figure()
            fig.add_scatter(x=fpr, y=tpr, mode='lines+markers', name=feature)
            fig.update_layout(
                autosize=False,
                width=600,
                height=500,
                title=f"ROC Curve - {feature}",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                showlegend=True
            )
            fig.add_annotation(
                x=0.6, y=0.2,
                text=f"AUC: {roc_auc:.3f}",
                showarrow=False,
                font=dict(size=12, color="black")
            )
            fig.show()

    # Sort features by AUC (highest first)
    auc_scores_sorted = sorted(auc_scores, key=lambda x: x[1], reverse=True)
  
    res = ""
    for name, score in auc_scores_sorted:
        res += name + " -- > AUC: " + str(score) + "\n"

    print_log("\n Ranking of Features by ROC-AUC:\n" + res)
    
    return auc_scores_sorted

def generate_cov_matrix(X, show_img=True):
    """
    Computes the covariance matrix of the dataset and identifies highly correlated feature pairs.

    Parameters:
        X (pd.DataFrame): Feature dataset.
        show_img (bool, optional): Whether to visualize the covariance matrix using Plotly. 
                                   Defaults to True.

    Returns:
        tuple: 
            - pd.DataFrame: Covariance matrix with feature names as row/column labels.
            - list: A sorted list of highly correlated feature pairs (absolute correlation > 0.8).
    """

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
        fig.write_image(f"corr_mat.png")

    corrMat = pd.DataFrame(corrMat, index=feature_names, columns=feature_names)

    # Find feature pairs with correlation above threshold (0.8)
    high_corr_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):  # Only upper triangle
            if abs(corrMat.iloc[i, j]) > 0.8:
                high_corr_pairs.append((feature_names[i], feature_names[j], corrMat.iloc[i, j]))

    # Sort by absolute correlation value (descending order)
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    res = ""
    for pair in high_corr_pairs:
        res += f"Features: {pair[0]} & {pair[1]} --> Correlation: {pair[2]:.2f}\n"
    
    print_log(res)

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
    
    #print_log(f"Removing {num_features_to_remove} worst features: {worst_features}")
    
    return X.drop(columns=worst_features)


#function to perform PCA and check which components to keep based on kaiser test
def analyse_pca(X, feature_selection_type, show_img=True):
    X  = StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(X)

    #PCA eigenvalues/Explained variance
    #print_log("PCA eigenvalues/Explained variance:\n")
    #print_log(pca.explained_variance_)
    #print_log("Sum of eigenvalues="+str(np.sum(pca.explained_variance_)))
    
    #PCA eigenvectors/Principal components
    #print_log("PCA eigenvectors/Principal components")
    W=pca.components_.T
    #print_log(W)

    print_log("The main PC contributes to "+str(pca.explained_variance_[0]**2/(pca.explained_variance_[0]**2+pca.explained_variance_[1]**2)*100)+"% of the variance.")
    
    #Kaiser test (quantos componentes têm eigenvalue acima de 1)
    optimal_components =sum(pca.explained_variance_>1)

    print_log("Optimal number of components according to Kaiser: "+str(optimal_components))

    #visualizar kaiser test
    if show_img:
        fig = px.scatter(x=np.arange(1,len(pca.explained_variance_)+1,1), y=pca.explained_variance_,
                    labels=dict(x="PC",y="Explained Variance"))
        fig.add_hline(y=1,line_width=3, line_dash="dash", line_color="red")
        fig.update_traces(marker_size=10)
        fig.show()
        fig.write_image(f"kaiser_test_{feature_selection_type}.png")

    print_log("Variance (%) retained accourding to Kaiser: "+str(pca.explained_variance_[0]**2/(np.sum(pca.explained_variance_**2))*100))
    print_log("Variance (%) retained accourding to Scree: "+str(np.sum(pca.explained_variance_[0:6]**2)/(np.sum(pca.explained_variance_**2))*100))

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


