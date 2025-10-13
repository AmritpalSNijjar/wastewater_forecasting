from abc import ABC, abstractmethod
import pandas as pd

class N2OModel(ABC):

    target_features = ["N2O", "N2O_lead1", "N2O_lead2"]
    
    def __init__(self, required_features):
        self.required_features = required_features

    def _dp_transform(self, X):
        """
        Takes a datapoint of the standard form, and returns the datapoint with only the necessary columns for prediction.

        Args:
            X (DataFrame): DataFrame of datapoints in the standard form.

        Returns:
            X (DataFrame): DataFrame containing only the required columns..
        """

        X = X[self.required_features].copy()
        
        return X

    def get_required_features(self):
        return self.required_features

    @abstractmethod
    def fit(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass
