import os
import sys

root_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from utils.data_loaders import cv_splits_loader

from sklearn.metrics import root_mean_squared_error as RMSE
import pandas as pd

def model_eval(model):

    train_errors = []
    val_errors = []

    for i in range(1, 8): # 8 data splits
        train_df, val_df = cv_splits_loader(split = i)

        train_pred = model.predict(train_df)
        train_error = RMSE(train_df[['N2O', 'N2O_lead1', 'N2O_lead2']], train_pred[['N2O', 'N2O_lead1', 'N2O_lead2']])/len(train_df)
        train_errors.append(train_error)

        val_pred = model.predict(val_df)
        val_error = RMSE(val_df[['N2O', 'N2O_lead1', 'N2O_lead2']], val_pred[['N2O', 'N2O_lead1', 'N2O_lead2']])/len(val_df)
        val_errors.append(val_error)
    
    return train_errors, val_errors
