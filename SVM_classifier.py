from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import plotly.express as px
from tqdm import tqdm

class SvmClassifier:
    def __init__(self):
        self.C = None
        self.classifier = None
        self.classifier_label = "SVM Classifier"

    def svm_analysis(self, X, y, C_values=np.logspace(-2, 2, 5), n_runs=5, test_size=0.3, view=False, random_seed=42):
        """
        Avalia o desempenho do SVM para diferentes valores de C e várias partições dos dados.

        Args:
            X (pd.DataFrame): Dados de entrada (features).
            y (pd.Series): Rótulos binários.
            C_values (array-like): Valores de C a testar.
            n_runs (int): Número de partições aleatórias a testar.
            test_size (float): Proporção do conjunto de teste.
            random_seed (int): Seed para reprodutibilidade.

        Retorna:
            dict: Resultados com erros médios, desvios e melhor C.
        """
        np.random.seed(random_seed)
        err_mat = np.zeros((n_runs, len(C_values)))

        total = n_runs * len(C_values)
        pbar = tqdm(total=total, desc="SVM Training", unit="model")

        for r in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

            for i, C in enumerate(C_values):
                svm = SVC(C=C, kernel='rbf', probability=True)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                err_mat[r, i] = (1 - acc) * 100
                pbar.update(1)

        pbar.close()

        avg_error = np.mean(err_mat, axis=0)
        std_error = np.std(err_mat, axis=0)
        opt_C = C_values[np.argmin(avg_error)]

        if view:
            fig = px.scatter(x=np.log10(C_values), y=avg_error, error_y=std_error,
                             labels={"x": "log10(C)", "y": "Average Error (%)"},
                             title="Average Classification Error ± Std for Different C (SVM)")
            fig.update_traces(marker=dict(size=8, color="DarkGreen"))
            fig.update_layout(font=dict(size=16))
            fig.show()

        print(f"Best C = {opt_C}")
        print(f"Minimum Average Error = {avg_error[np.argmin(avg_error)]:.2f}%")

        self.C = opt_C

        return {
            "error_matrix": err_mat,
            "average_error": avg_error,
            "std_error": std_error,
            "best_C": opt_C
        }

    def train(self, X_train, Y_train):
        # Get optimal C first
        self.svm_analysis(X_train, Y_train)
        self.classifier = SVC(C=self.C, kernel='rbf', probability=True)
        self.classifier.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    def objective_function(self, X_test):
        return self.classifier.predict_proba(X_test)[:, 1]
