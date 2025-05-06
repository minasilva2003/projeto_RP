import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt

def calculate_sensitivity(y_true, y_pred, positive_label=1):
    """
    Calculates the sensitivity (True Positive Rate).

    Args:
        y_true (np.array): True class labels.
        y_pred (np.array): Predicted class labels.
        positive_label: The label considered as the positive class.

    Returns:
        float: The sensitivity score.
    """
    true_positives = np.sum((y_true == positive_label) & (y_pred == positive_label))
    false_negatives = np.sum((y_true == positive_label) & (y_pred != positive_label))
    if (true_positives + false_negatives) == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)

def calculate_specificity(y_true, y_pred, negative_label=0):
    """
    Calculates the specificity (True Negative Rate).

    Args:
        y_true (np.array): True class labels.
        y_pred (np.array): Predicted class labels.
        negative_label: The label considered as the negative class.

    Returns:
        float: The specificity score.
    """
    true_negatives = np.sum((y_true == negative_label) & (y_pred == negative_label))
    false_positives = np.sum((y_true == negative_label) & (y_pred != negative_label))
    if (true_negatives + false_positives) == 0:
        return 0.0
    return true_negatives / (true_negatives + false_positives)



def cross_validate_classifier(X, y, classifier, n_folds=5, view=True):
    """
    Executa validação cruzada com K folds para um classificador dado.

    Parâmetros:
        X (pd.DataFrame): Conjunto de features.
        y (pd.Series): Labels correspondentes (0 ou 1).
        classifier_fn (class): Classifier class
        n_folds (int): Número de folds para cross-validation. Default = 5.
        label (str): Nome do classificador (para display).

    Retorna:
        tuple: (accuracy média, especificidade média)
    """
  
   
    # Criar vetor de índices embaralhados
    indices = np.arange(len(X))
    fold_size = len(X) // n_folds
    np.random.seed(42)  # Garantir reprodutibilidade
    np.random.shuffle(indices)

    # Inicializar listas para guardar resultados de cada fold
    accuracy_scores = []
    specificity_scores = []
    auc_scores = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    print(f"\n*** Cross-validation for {classifier.classifier_label} ***")

    for i in range(n_folds):
        # Definir os índices para o fold de validação atual
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(X)  # último fold pode ter tamanho ligeiramente maior

        val_indices = indices[start:end]  # dados para validação
        train_indices = np.concatenate([indices[:start], indices[end:]])  # dados para treino

        # Separar os dados de treino e validação com base nos índices definidos
        X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
        y_train, y_val = y.iloc[train_indices], y.iloc[val_indices]

        # Executar o classificador no fold atual
        classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_val)

        # Calcular métricas de desempenho
        acc = accuracy_score(y_val, y_pred)
        spec = calculate_specificity(y_val, y_pred)

        # Armazenar resultados
        accuracy_scores.append(acc)
        specificity_scores.append(spec)

        # Calcular a curva ROC
        y_scores = classifier.objective_function(X_val)
        fpr, tpr, _ = roc_curve(y_val, y_scores)
        roc_auc = auc(fpr, tpr)
        auc_scores.append(roc_auc)

        # Interpolação para média de curvas ROC
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)


    # Mostrar resultados agregados (média e desvio padrão)
    print(f"Mean Accuracy for {classifier.classifier_label}: {np.mean(accuracy_scores):.3f} ± {np.std(accuracy_scores):.3f}")
    print(f"Mean Specificity {classifier.classifier_label}: {np.mean(specificity_scores):.3f} ± {np.std(specificity_scores):.3f}")

    # Plot curva ROC média (se houver score_fn)
    if len(auc_scores) > 0 and view:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=f'{classifier.classifier_label} (AUC = {mean_auc:.2f})', lw=2)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Mean ROC Curve - {classifier.classifier_label}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return np.mean(accuracy_scores), np.mean(specificity_scores), mean_auc
        
    return np.mean(accuracy_scores), np.mean(specificity_scores), None
