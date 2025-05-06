from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score  # sensitivity = recall for positive class
import numpy as np
import plotly.express as px
from sklearn.metrics import accuracy_score

class KnnClassifier:
    def __init__(self):

        self.k = None
        self.classifier = None
        self.classifier_label = "KNN Classifier"

    def knn_analysis(self, X, y, k_values=np.arange(1, 16, 2), n_runs=5, test_size=0.3, view=False, random_seed=42):
        """
        Avalia o desempenho do k-NN para diferentes valores de k e várias partições dos dados.
        
        Args:
            X (pd.DataFrame): Dados de entrada (features).
            y (pd.Series): Rótulos binários.
            k_values (array-like): Valores de k a testar.
            n_runs (int): Número de partições aleatórias a testar.
            test_size (float): Proporção do conjunto de teste.
            random_seed (int): Seed para reprodutibilidade.
            
        Retorna:
            dict: Resultados com erros médios, desvios e melhor k.
        """
        np.random.seed(random_seed)
        err_mat = np.zeros((n_runs, len(k_values)))

        for r in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            for i, k in enumerate(k_values):
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                err_mat[r, i] = (1 - acc) * 100  # store error in %

        avg_error = np.mean(err_mat, axis=0)
        std_error = np.std(err_mat, axis=0)
        opt_k = k_values[np.argmin(avg_error)]

        # Plot
        if view:
            fig = px.scatter(x=k_values, y=avg_error, error_y=std_error,
                            labels={"x": "k", "y": "Average Error (%)"},
                            title="Average Classification Error ± Std for Different k (k-NN)")
            fig.update_traces(marker=dict(size=8, color="RebeccaPurple"))
            fig.update_layout(font=dict(size=16))
            fig.show()

        print(f"Best k = {opt_k}")
        print(f"Minimum Average Error = {avg_error[np.argmin(avg_error)]:.2f}%")

        self.k = opt_k

        return {
            "error_matrix": err_mat,
            "average_error": avg_error,
            "std_error": std_error,
            "best_k": opt_k
        }


    def train(self, X_train, Y_train):

        #get optimal k first
        self.knn_analysis(X_train, Y_train)

        self.classifier = KNeighborsClassifier(n_neighbors=self.k)
        self.classifier.fit(X_train, Y_train)
        
        
        
    def predict(self, X_test):
        Y_pred = self.classifier.predict(X_test)
        return Y_pred

    def objective_function(self, X_test):
        y_prob = self.classifier.predict_proba(X_test)[:, 1]
        return y_prob


