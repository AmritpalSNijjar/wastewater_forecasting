from .base import N2OModel
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import GammaRegressor

class ChainedAutoLinearRegression(N2OModel):
    """
    A forecasting model based on linear regression. This model predicts the next value of N2O as a linear combination of the last `w` parameters,
    where `w`, the window size, is a hyperparameter. 
    """
    
    def __init__(self, window_size = 3, regressor = "lin"):
        
        assert 0 < window_size < 21, "Window size must be between 1 and 20."
        self.window_size = window_size
        
        required_features = []
        
        for i in range(1, window_size + 1):
            required_features.append(f"N2O_lag{i}")

        super().__init__(required_features = required_features)

        self.regressor = regressor
        
        if self.regressor == "gam":
            self.model = GammaRegressor()
            
        if self.regressor == "lin":
            self.model = LinearRegression()

    def fit(self, X):
        epsilon = 1e-6
        y = X[[self.target_features[0]]] + epsilon
        X = self._dp_transform(X)

        self.model.fit(X, y)

    def predict(self, X):
        """
        Takes a standard datapoint, and predicts recursivley predicts the next values of N2O as a linear function of the last 
        `self.window_size` values.

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
            
            y[lead] = pd.Series(self.model.predict(X).flatten(), x_index)

            # Drop the oldest lag, as it is not necessary for making the next prediction.
            X.drop(furthest_lag, axis=1, inplace=True)

            # Shift all lags back by one in X.
            X.rename(columns=rename_dict, inplace=True)
            
            # Add the new prediction as the lag1 column in X
            X["N2O_lag1"] = y[lead]

            # Shuffle columns to the correct order for linear regression.
            X = X[self.required_features]

        return y[self.target_features]

class MultiOutputLinearRegression(N2OModel):
    """
    A forecasting model based on linear regression. This model predicts the next 3 values of N2O concentration using 3 separate linear regression models. 
    """
    
    def __init__(self, window_size = 3, regressor = "lin"):
        
        assert 0 < window_size < 21, "Window size must be between 1 and 20."
        self.window_size = window_size
        
        required_features = []
        
        for i in range(1, window_size + 1):
            required_features.append(f"N2O_lag{i}")

        super().__init__(required_features = required_features)

        
        self.regressor = regressor
        
        if self.regressor == "gam":
            self.n2o_model = GammaRegressor()
            self.n2o_lead1_model = GammaRegressor()
            self.n2o_lead2_model = GammaRegressor()
            
        if self.regressor == "lin":
            self.n2o_model = LinearRegression()
            self.n2o_lead1_model = LinearRegression()
            self.n2o_lead2_model = LinearRegression()
        

    def fit(self, X):
        epsilon = 1e-6
        
        y_n2o = X[[self.target_features[0]]] + epsilon
        y_n2o_lead1 = X[[self.target_features[1]]] + epsilon
        y_n2o_lead2 = X[[self.target_features[2]]] + epsilon
        X = self._dp_transform(X)

        self.n2o_model.fit(X, y_n2o)
        self.n2o_lead1_model.fit(X, y_n2o_lead1)
        self.n2o_lead2_model.fit(X, y_n2o_lead2)

    def predict(self, X):
        """
        Takes a standard datapoint, and predicts the next values of 3 N2O values using 3 separate linear regression models.

        Args:
            X (DataFrame): DataFrame of datapoints in the standard form.

        Returns:
            y (DataFrame): DataFrame with the predicted values of 'N2O', 'N2O_lead1', and 'N2O_lead2'.
        """

        X = self._dp_transform(X)

        x_index = X.index
        
        y = pd.DataFrame(index = x_index)

        leads = ["N2O", "N2O_lead1", "N2O_lead2"]
        models = [self.n2o_model, self.n2o_lead1_model, self.n2o_lead2_model]
        
        for lead, model in zip(leads, models):
            y[lead] = pd.Series(model.predict(X).flatten(), index=x_index)

        return y[self.target_features]

class VectorAutoRegression(N2OModel):
    """
    A forecasting model based on vector auto regression. The model uses lags of N2O, NH4, and NO3 upto `w` values into the past. 
    Separate linear regression models are fit to predict the next values of N2O, NH4, and NO3, and these model predictions 
    are chained together iteratively to predict the next three values of N2O. `w`, the window size, is a hyperparameter. 
    """
    
    def __init__(self, window_size = 3, regressor = "lin"):
        
        assert 0 < window_size < 21, "Window size must be between 1 and 20."
        self.window_size = window_size
        
        required_features = []
        
        for i in range(1, window_size + 1):
            required_features.append(f"N2O_lag{i}")
            required_features.append(f"NH4_lag{i}")
            required_features.append(f"NO3_lag{i}")

        super().__init__(required_features = required_features)

        self.regressor = regressor
        
        if self.regressor == "gam":
            self.N2O_model = GammaRegressor()
            self.NH4_model = GammaRegressor()
            self.NO3_model = GammaRegressor()

        if self.regressor == "lin":
            self.N2O_model = LinearRegression()
            self.NH4_model = LinearRegression()
            self.NO3_model = LinearRegression()      

    def fit(self, X):

        epsilon = 1e-6
        
        y_n2o = (X[['N2O']] + epsilon).values.ravel()
        y_no3 = (X[['NO3']] + epsilon).values.ravel()
        y_nh4 = (X[['NH4']] + epsilon).values.ravel()
        
        X = self._dp_transform(X)

        self.N2O_model.fit(X, y_n2o)
        self.NO3_model.fit(X, y_no3)
        self.NH4_model.fit(X, y_nh4)

    def predict(self, X):
        """
        Takes a standard datapoint, and predicts recursivley predicts the next values of N2O, NH4, and NO3 each as a separate 
        linear function of the last `self.window_size` values of all variables. Uses these three linear models to iteratively
        predict the next three values of N2O.

        Args:
            X (DataFrame): DataFrame of datapoints in the standard form.

        Returns:
            y (DataFrame): DataFrame with the predicted values of 'N2O', 'N2O_lead1', and 'N2O_lead2'.
        """

        X = self._dp_transform(X)

        x_index = X.index
        
        y = pd.DataFrame(index = x_index)

        furthest_lags = [f'{variable}_lag{self.window_size}' for variable in ('N2O', 'NH4', 'NO3')]

        rename_dict = {}
        for i in range(1, self.window_size):
            rename_dict[f"N2O_lag{i}"] = f"N2O_lag{i + 1}"
            rename_dict[f"NO3_lag{i}"] = f"NO3_lag{i + 1}"
            rename_dict[f"NH4_lag{i}"] = f"NH4_lag{i + 1}"

        for lead in self.target_features:

            y[lead] = pd.Series(self.N2O_model.predict(X).flatten(), x_index)
            
            nh4_pred = pd.Series(self.NH4_model.predict(X).flatten(), x_index)
            no3_pred = pd.Series(self.NO3_model.predict(X).flatten(), x_index)

            # Drop the oldest lags for each variable, as they is not necessary for making the next prediction.
            X.drop(furthest_lags, axis=1, inplace=True)

            # Shift all lags back by one in X.
            X.rename(columns=rename_dict, inplace=True)
            
            # Add the new predictions as the lag1 columns in X for each variable
            X["N2O_lag1"] = y[lead]
            X["NH4_lag1"] = nh4_pred
            X["NO3_lag1"] = no3_pred

            # Shuffle columns to the correct order for linear regression.
            X = X[self.required_features]

        return y[self.target_features]


