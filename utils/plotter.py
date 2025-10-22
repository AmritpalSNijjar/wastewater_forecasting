import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_pred(model, datapoint, output_path = None):
    """
    Plots the true and predicted N2O values for a given model and datapoint.

    Args:
        model (Model): Model to use in prediction.
        datapoint (DataFrame): DataFrame which contains a single row in the standard form, 
                               where any of the `N2O_lag#`, `NO3_lag#`, and `NH4_lag#` 
                               columns may be used for prediction.
        output_path (str): Location to save file if desired.
    """

    current_time = datapoint.index[0]
    
    n_past = 20
    n_future = 2

    time_array = pd.date_range(end = current_time + pd.Timedelta(minutes=n_future*10), periods=n_past + 1 + n_future, freq='10min')

    truth_upto = datapoint[[f'N2O_lag{lag}' for lag in range(20, 0, -1)]].to_numpy()[0]
    truth_to_compare = datapoint[['N2O', 'N2O_lead1', 'N2O_lead2']].to_numpy()[0]
    pred = model.predict(datapoint).to_numpy()[0]

    plt.plot(time_array, np.concatenate((truth_upto, truth_to_compare)), linestyle=':', color = 'r', linewidth = 3, label = 'Truth')
    plt.plot(time_array, np.concatenate((truth_upto, pred)), color = 'maroon', linewidth = 2.7, label = 'Prediction')
    plt.plot(time_array[:20], truth_upto, color = 'r',linewidth = 3)
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator())
    plt.gca().xaxis.set_minor_locator(mdates.MinuteLocator(byminute=range(0,60,10)))
    plt.xticks(rotation=45)
    plt.xlabel('Time')
    plt.ylabel('N2O') # add units!!
    plt.title('N2O Prediction vs Truth') 
    plt.legend()
    
    if output_path:
        pass
