import numpy as np
import pandas as pd
import sklearn.metrics as skm


class RegressionMetrics:
    def __init__(self, y_true, y_pred, name=None):
        """Initialize the class with the true and predicted values
        Args:
            y_true (np.array): Values that are true
            y_pred (np.array): Predicted values
        """
        self.y_true = y_true
        self.y_pred = y_pred
        if name is None:
            self.name = "Regression Metrics"
        else:
            self.name = name

    @property
    def r2(self):
        return skm.r2_score(self.y_true, self.y_pred)

    @property
    def mse(self):
        return skm.mean_squared_error(self.y_true, self.y_pred)

    @property
    def rmse(self):
        return np.sqrt(self.mse)

    @property
    def mae(self):
        return skm.mean_absolute_error(self.y_true, self.y_pred)

    @property
    def mape(self):
        return np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100

    @property
    def rmspe(self):
        return (
            np.sqrt(np.mean(np.square((self.y_true - self.y_pred) / self.y_true))) * 100
        )

    @property
    def max_error(self):
        return skm.max_error(self.y_true, self.y_pred)

    @property
    def explained_variance_score(self):
        return skm.explained_variance_score(self.y_true, self.y_pred)

    def regression_report(self):
        return pd.DataFrame(
            {
                "R2": [self.r2],
                "MSE": [self.mse],
                "RMSE": [self.rmse],
                "MAE": [self.mae],
                "MAPE": [self.mape],
                "RMSPE": [self.rmspe],
                "Max Error": [self.max_error],
                "Explained Variance Score": [self.explained_variance_score],
            }
        )

    def __call__(self) -> pd.DataFrame:
        """Call self as a function to get the report."""
        report = self.regression_report()
        report.index = [self.name]
        return report
