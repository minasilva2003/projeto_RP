import numpy as np

def drop_categorical_variables(X): 
    categorical_columns = X.select_dtypes(include=['object', 'category']).columns
    print(f"Filtering categoric variables: {categorical_columns[1:]}")
    return X.drop(columns=categorical_columns)

def drop_binary_variables(X):
    binary_columns = [col for col in X.columns if X[col].nunique() == 2 and X[col].dropna().isin([0, 1]).all()]
    print(f"Filtering binary columns: {binary_columns}")
    return X.drop(columns=binary_columns)

def normalize_features(X):
    return ((X-np.mean(X,axis=0))/np.std(X,axis=0))