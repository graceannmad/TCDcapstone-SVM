import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class OCSVM:
    """
    Simple One-Class SVM anomaly detector implemented with scikit library SVM.
    """

    def __init__(self, kernel="rbf", nu=0.05, gamma="scale"):
        """
        Args:
            kernel (str): Kernel type ('rbf', 'linear', 'poly', etc.). [rbf = Radial basis function]
            nu (float): An upper bound on the fraction of anomalies.
            gamma (str or float): Kernel coefficient.
        """
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.scaler = StandardScaler() #this will standardize the data before feeding it to the SVM
        self.model = OneClassSVM(kernel=self.kernel, nu=self.nu, gamma=self.gamma)

    def fit(self, X_train):
        """Fit the One-Class SVM to training data. (Training)"""
        X_scaled = self.scaler.fit_transform(X_train) #standardization from above
        self.model.fit(X_scaled)

    def predict(self, X_test):
        """Predict normal (+1) or anomaly (-1) for test data. (Run inference on the test data)"""
        X_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_scaled)

    def score_samples(self, X_test):
        """Return anomaly scores (higher = more normal)."""
        X_scaled = self.scaler.transform(X_test)
        return self.model.score_samples(X_scaled)
