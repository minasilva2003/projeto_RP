import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from SVM_classifier import SvmClassifier

# Preprocessing + analysis functions
from pre_process_funcs import normalize_features, drop_categorical_variables, drop_binary_variables
from feature_analysis_funcs import (
    kruskal_wallis, generate_cov_matrix, remove_worst_features,
    analyse_pca
)

# -------------------------------------
# 0. Load dataset
url_dataset = pd.read_csv("dataset.csv")
X = url_dataset.drop(columns=["label"])
Y = url_dataset["label"]
print("Initial shape:", X.shape)

# -------------------------------------
# 1. Drop categorical and binary features
X = drop_categorical_variables(X)
X = drop_binary_variables(X)
print("After dropping categorical and binary:", X.shape)

# -------------------------------------
# 2. Normalize features
X = normalize_features(X)

# -------------------------------------
# 3. Correlation matrix & remove known redundant features
_, high_corr_pairs = generate_cov_matrix(X, show_img=False)
X = X.drop(columns=["DomainTitleMatchScore", "NoOfLettersInURL", "NoOfDegitsInURL"], errors="ignore")
print("After correlation-based removal:", X.shape)

# -------------------------------------
# 4. Feature selection with Kruskal-Wallis
kw_ranking = kruskal_wallis(X, Y)
X = remove_worst_features(X, kw_ranking, percentage=0.3)
print("After removing bottom 30% KW features:", X.shape)

# -------------------------------------
# 5. PCA reduction

from sklearn.model_selection import train_test_split

# Randomly sample a portion of the dataset (e.g., 20% of the data)
X_subset, _, _, _ = train_test_split(X, X, test_size=0.95, random_state=42)

# Now perform PCA on the subset
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)  # Adjust for your desired number of components
X_pca = pca.fit_transform(X_subset)

print(f"PCA on subset shape: {X_pca.shape}")


X_pca = analyse_pca(X, show_img=False)
print("PCA-reduced shape:", X_pca.shape)

# -------------------------------------
# 6. Train SVM on small sample for tuning
X_small, _, y_small, _ = train_test_split(
    X_pca, Y, train_size=10000, stratify=Y, random_state=42
)

svm = SvmClassifier()
results = svm.svm_analysis(X_small, y_small, C_values=np.logspace(-2, 2, 5), n_runs=5, view=True)

# -------------------------------------
# 7. Final train on full dataset using best C
svm.train(X_pca, Y)

