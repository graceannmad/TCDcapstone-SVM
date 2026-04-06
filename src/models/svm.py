from src.models.standard_smo import SMO

class SVM:
    """
    SVM implementation
    """
    def __init__(self, kernel, gamma, C):
        """
        Args:
            kernel (function)
            nu (float): An upper bound on the fraction of anomalies.
            gamma (str or float): Kernel coefficient.
        """
        self.kernel = kernel
        self.gamma = gamma
        self.C = C

        #not yet used - need to train on data first
        self.alpha, self.beta = 0, 0
        self.X_train, self.y_train = 0, 0
        self.fitted = False

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        smo = SMO(self.kernel, X_train, y_train, self.C)
        self.alpha, self.beta = smo.run_smo()

        self.fitted = True

    def pred_one(self, x):
        result = 0
        # f(x) = SUM_{i=1}^n [alpha_i * y_i * K(x_i, x)]
        for i in range(len(self.alpha)):
            result += self.alpha[i] * self.y_train[i] * self.kernel(self.X_train[i], x)
        result += self.beta
        return 1 if result >= 0 else -1

    def predict(self, X_test):
        """Predict normal (+1) or anomaly (-1) for test data. (Run inference on the test data)"""
        if not self.fitted:
            print("You must train your SVM before predicting with it!")
            return

        n, _ = X_test.shape
        preds = [0]*n
        for i in range(n):
            preds[i] = self.pred_one(X_test[i])

        return preds