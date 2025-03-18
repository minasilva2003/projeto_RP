import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import MinMaxScaler

def drop_categorical_variables(dataset, features): 
    categorical_columns = dataset.variables[dataset.variables["type"] == "Categorical"]["name"].tolist()
    print(f"Filtering categoric variables: {categorical_columns[1:]}")
    return features.drop(columns=categorical_columns[1:])

def drop_binary_variables(features):
    integer_columns = features.select_dtypes(include=['int64', 'int32']).columns
    binary_columns = [col for col in integer_columns if features[col].dropna().isin([0, 1]).all()]
    print(f"Filtering binary columns: {binary_columns}")
    return features.drop(columns=binary_columns)

def normalize_features(features):
    scaler = MinMaxScaler()
    normalized_array = scaler.fit_transform(features)  # Normalize the values
    normalized_df = pd.DataFrame(normalized_array, columns=features.columns)  # Convert back to DataFrame
    print("Normalizing dataset...")
    return normalized_df
#________________________________________________________________________________

#0. ler dataset
url_dataset = fetch_ucirepo(id=967) 

X = url_dataset.data.features 
y = url_dataset.data.targets 
  
#print(url_dataset.metadata) 
#print(url_dataset.variables) 

#_________________________________________________________________________________

#1. remover categóricas e binárias
print(X.shape)
X = drop_categorical_variables(url_dataset, X)
print(X.shape)
X = drop_binary_variables(X)
print(X.shape)

#_________________________________________________________________________________

#2. limpar dataset
#quando se faz url_dataset.variables ele supostamente diz que nenhuma coluna tem missing data portanto não sei se isto é necessário

#__________________________________________________________________________________

#3. normalizar/scaling
X = normalize_features(X)
print(X.head())

#4. kriskow walix para feature selection
#5. PCA e LDA sobre as selected features para criar novas features