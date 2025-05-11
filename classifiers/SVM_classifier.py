from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import plotly.express as px
from tqdm import tqdm
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

class SvmClassifier:
    def __init__(self, kernel_function, data_info):
        self.C = None
        self.classifier = None
        self.classifier_label = "SVM_Classifier"
        self.kernel_function = kernel_function
        self.objective_function = None
        self.data_info = data_info

    def svm_analysis(self, X, y, C_values=np.logspace(-2, 2, 5), n_runs=5, test_size=0.3, view=False, random_seed=42):
        """
        Evaluates the performance of SVM for different values of C and multiple data partitions.

        Args:
            X (pd.DataFrame): Input data (features).
            y (pd.Series): Binary labels.
            C_values (array-like): Values of C to test.
            n_runs (int): Number of random partitions to test.
            test_size (float): Proportion of the test set.
            random_seed (int): Seed for reproducibility.

        Returns:
            dict: Results with average errors, standard deviations, and the best C.
        """
        np.random.seed(random_seed)
        err_mat = np.zeros((n_runs, len(C_values)))

        total = n_runs * len(C_values)
        pbar = tqdm(total=total, desc="SVM Training", unit="model")

        # Initialize the scaler
        scaler = MinMaxScaler()

        for r in range(n_runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

            X_train = pd.DataFrame(scaler.fit_transform(X_train))
            X_test = pd.DataFrame(scaler.transform(X_test))

            for i, C in enumerate(C_values):
                svm = SVC(C=C, kernel=self.kernel_function, probability=False, max_iter=1000)
                svm.fit(X_train, y_train)
                y_pred = svm.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                err_mat[r, i] = (1 - acc) * 100
                pbar.update(1)

        pbar.close()

        avg_error = np.mean(err_mat, axis=0)
        std_error = np.std(err_mat, axis=0)
        opt_C = C_values[np.argmin(avg_error)]

        
        fig = px.scatter(x=np.log10(C_values), y=avg_error, error_y=std_error,
                            labels={"x": "log10(C)", "y": "Average Error (%)"},
                            title="Average Classification Error ± Std for Different C (SVM)")
        fig.update_traces(marker=dict(size=8, color="DarkGreen"))
        fig.update_layout(font=dict(size=16))
        fig.write_image(f"svm_training/img_{self.data_info}.png")
        
        if view:
            fig.show()

        print_log(f"Best C = {opt_C}")
        print_log(f"Minimum Average Error = {avg_error[np.argmin(avg_error)]:.2f}%")

        self.C = opt_C

        return [C_values, avg_error, std_error]
    
    def train(self, X_train, Y_train):
        self.classifier = SVC(C=self.C, kernel=self.kernel_function, probability=False)
        self.classifier.fit(X_train, Y_train)

    def predict(self, X_test):
        return self.classifier.predict(X_test)

    