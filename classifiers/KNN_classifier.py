from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score  # sensitivity = recall for positive class
import numpy as np
import plotly.express as px
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

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


class KnnClassifier:
    def __init__(self, data_info):

        self.k = None
        self.classifier = None
        self.classifier_label = "KNN_Classifier"
        self.data_info = data_info

    def knn_analysis(self, X, y, k_values=np.arange(1, 16, 2), n_runs=5, test_size=0.3, view=False, random_seed=42):
        """
        Evaluates the performance of k-NN for different values of k and multiple data partitions.
        
        Args:
            X (pd.DataFrame): Input data (features).
            y (pd.Series): Binary labels.
            k_values (array-like): Values of k to test.
            n_runs (int): Number of random partitions to test.
            test_size (float): Proportion of the test set.
            random_seed (int): Seed for reproducibility.
            
        Returns:
            dict: Results with average errors, standard deviations, and the best k.
        """
        np.random.seed(random_seed)
        err_mat = np.zeros((n_runs, len(k_values)))

        total = n_runs * len(k_values)
        pbar = tqdm(total=total, desc="Knn Training", unit="model")

        # Initialize the scaler
        scaler = MinMaxScaler()

        for r in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

            X_train = pd.DataFrame(scaler.fit_transform(X_train))
            X_test = pd.DataFrame(scaler.transform(X_test))

            for i, k in enumerate(k_values):
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                err_mat[r, i] = (1 - acc) * 100  # store error in %
                pbar.update(1)

        pbar.close()

        avg_error = np.mean(err_mat, axis=0)
        std_error = np.std(err_mat, axis=0)
        opt_k = k_values[np.argmin(avg_error)]

        # Plot
        
        fig = px.scatter(x=k_values, y=avg_error, error_y=std_error,
                        labels={"x": "k", "y": "Average Error (%)"},
                        title="Average Classification Error ± Std for Different k (k-NN)")
        fig.update_traces(marker=dict(size=8, color="RebeccaPurple"))
        fig.update_layout(font=dict(size=16))
        fig.write_image(f"knn_training/img_{self.data_info}.png")
        if view:
            fig.show()

        print_log(f"Best k = {opt_k}")
        print_log(f"Minimum Average Error = {avg_error[np.argmin(avg_error)]:.2f}%")

        self.k = opt_k
        """
        return {
            "error_matrix": err_mat,
            "average_error": avg_error,
            "std_error": std_error,
            "best_k": opt_k
        }
        """
        return [k_values, avg_error, std_error]


    def train(self, X_train, Y_train):

        self.classifier = KNeighborsClassifier(n_neighbors=self.k)
        self.classifier.fit(X_train, Y_train)
        
        
        
    def predict(self, X_test):
        Y_pred = self.classifier.predict(X_test)
        return Y_pred

    def objective_function(self, X_test):
        y_prob = self.classifier.predict_proba(X_test)[:, 1]
        return y_prob


