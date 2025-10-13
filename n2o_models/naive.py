from base import N2OModel
import pandas as pd

class NaivePersistence(N2OModel):
    """
    A simple baseline forecasting model that predicts future N2O values as the most recently observed (lagged) value.
    """
    
    def __init__(self):
        super().__init__(required_features = ['N2O_lag1'])

    def fit(self, X):
        pass

    def predict(self, X):
        """
        Takes a standard datapoint, and predicts the next 3 values of N2O as the last observed values of N2O.

        Args:
            X (DataFrame): DataFrame of datapoints in the standard form.

        Returns:
            y (DataFrame): DataFrame with the predicted values of 'N2O', 'N2O_lead1', and 'N2O_lead2'.
        """

        X = self._dp_transform(X)

        y = pd.DataFrame(index = X.index)

        y['N2O'] = X['N2O_lag1']
        y['N2O_lead1'] = X['N2O_lag1']
        y['N2O_lead2'] = X['N2O_lag1']
        
        return y[self.target_features]

class NaiveRollingMean(N2OModel):
    """
    A simple baseline forecasting model that predicts future N2O values
    as the rolling mean of the most recently observed lagged values.
    """
    
    def __init__(self, window_size = 3):
        
        assert 1 < window_size < 21, "Window size must be between 1 and 20."
        self.window_size = window_size
        
        required_features = []
        
        for i in range(1, window_size + 1):
            required_features.append(f"N2O_lag{i}")

        super().__init__(required_features = required_features)

    def fit(self, X):
        pass

    def predict(self, X):
        """
        Takes a standard datapoint, and predicts the next 3 values of N2O as rolling means of the most recent lagged values.

        Args:
            X (DataFrame): DataFrame of datapoints in the standard form.

        Returns:
            y (DataFrame): DataFrame with the predicted values of 'N2O', 'N2O_lead1', and 'N2O_lead2'.
        """

        X = self._dp_transform(X)

        y = pd.DataFrame(index = X.index)

        cols_to_mean = list(self.required_features)

        y['N2O'] = X[cols_to_mean].mean(axis=1)
        cols_to_mean = ['N2O'] + cols_to_mean[:-1]
        X['N2O'] = y['N2O']
        X = X[cols_to_mean]
        
        y['N2O_lead1'] = X[cols_to_mean].mean(axis=1)
        cols_to_mean = ['N2O_lead1'] + cols_to_mean[:-1]
        X['N2O_lead1'] = y['N2O_lead1']
        X = X[cols_to_mean]
        
        y['N2O_lead2'] = X[cols_to_mean].mean(axis=1)
        
        return y[self.target_features]