import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
from sklearn.neighbors import KNeighborsClassifier

def euclidean_objective_function(X_train, y_train, X_val):
    import numpy as np
    # Get class mean for phishing (label 1)
    phishing_mean = np.mean(X_train[y_train == 1], axis=0)
    return -np.linalg.norm(X_val - phishing_mean, axis=1)  # more negative = less confident



def mahalanobis_objective_function(X_train, y_train, X_val):
    """
    Returns negative Mahalanobis distance to phishing class mean (label = 1).
    Higher score = more phishing-like.
    """
    X_train_np = np.asarray(X_train)
    X_val_np = np.asarray(X_val)
    y_train_np = np.asarray(y_train)

    # Compute phishing class mean and covariance
    phishing_class = 1
    phishing_samples = X_train_np[y_train_np == phishing_class]
    phishing_mean = np.mean(phishing_samples, axis=0)
    cov = np.cov(phishing_samples, rowvar=False)
    inv_cov = inv(cov)

    # Score = -mahalanobis distance to phishing mean
    scores = [-mahalanobis(x, phishing_mean, inv_cov) for x in X_val_np]
    return np.array(scores)


def fisher_objective_function(X_train, y_train, X_val):
    """
    Returns projection scores on Fisher LDA direction.
    Higher score = more phishing-like.
    """
    X_train_np = np.asarray(X_train)
    y_train_np = np.asarray(y_train)
    X_val_np = np.asarray(X_val)

    # Compute class means and scatter matrices
    class0 = X_train_np[y_train_np == 0]
    class1 = X_train_np[y_train_np == 1]

    m0 = np.mean(class0, axis=0)
    m1 = np.mean(class1, axis=0)

    S0 = np.cov(class0, rowvar=False)
    S1 = np.cov(class1, rowvar=False)
    Sw = S0 + S1

    # Compute Fisher LDA direction
    w = np.linalg.inv(Sw).dot(m1 - m0)

    # Project validation data
    scores = X_val_np @ w  # projection values
    return scores


def knn_objective_function(X_train, Y_train, X_test):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    y_prob = knn.predict_proba(X_test)[:, 1]
    return y_prob