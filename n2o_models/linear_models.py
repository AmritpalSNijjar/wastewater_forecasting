from .base import N2OModel
import pandas as pd
from sklearn.linear_model import LinearRegression

class ChainedLinearRegression(N2OModel):
    """
    A forecasting model based on linear regression. This model predicts the next value of N2O as a linear combination of the last`w` parameters,
    where `w`, the window size, is a hyperparameter. 
    """
    
    def __init__(self, window_size = 3):
        
        assert 0 < window_size < 21, "Window size must be between 1 and 20."
        self.window_size = window_size
        
        required_features = []
        
        for i in range(1, window_size + 1):
            required_features.append(f"N2O_lag{i}")

        super().__init__(required_features = required_features)

        self.model = LinearRegression()

    def fit(self, X):
        y = X[[self.target_features[0]]]
        X = self._dp_transform(X)

        self.model.fit(X, y)

    def predict(self, X):
        """
        Takes a standard datapoint, and predicts recursivley predicts the next values of N2O as a linear function of the `self.window_size`
        values.

        Args:
            X (DataFrame): DataFrame of datapoints in the standard form.

        Returns:
            y (DataFrame): DataFrame with the predicted values of 'N2O', 'N2O_lead1', and 'N2O_lead2'.
        """

        X = self._dp_transform(X)

        x_index = X.index
        
        y = pd.DataFrame(index = x_index)

        furthest_lag = self.get_required_features()[-1]

        rename_dict = {}
        for i in range(1, self.window_size):
            rename_dict[f"N2O_lag{i}"] = f"N2O_lag{i + 1}"

        for lead in self.target_features:

            y[lead] = pd.Series(self.model.predict(X), x_index)

            # Drop the oldest lag, as it is not necessary for making the next prediction.
            X.drop(furthest_lag, axis=1, inplace=True)

            # Shift all lags back by one in X.
            X.rename(columns=rename_dict, inplace=True)
            
            # Add the new prediction as the lag1 column in X
            X["N2O_lag1"] = y[lead]

            # Shuffle columns to the correct order for linear regression.
            X = X[self.required_features]

        return y[self.target_features]
    