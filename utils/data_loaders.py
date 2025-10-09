import pandas as pd
from pathlib import Path

def cv_splits_loader(split = 1):
    """
    Load training and validation datasets for time series training, using the expanding window method.
    
    Args:
        split (int): Number of training splits to combine (1 – 8).
    
    Returns:
        (train_df, val_df) (DataFrame, DataFrame): Tuple consisting of training dataset with `split` consecutive datasets, and validation dataset.
    """

    assert 1 <= split <= 8, "split must be between 1 and 8"

    base_path = Path(__file__).resolve().parents[1] / "data" / "processed"

    train_dfs = [ pd.read_csv(base_path / f"wastewater_tank1_processed_split_{i}.csv", index_col=0, parse_dates=True) for i in range(1, split + 1)]
    train_df = pd.concat(train_dfs, axis=0)
    
    val_data_path = base_path / f"wastewater_tank1_processed_split_{split + 1}.csv"
    val_df = pd.read_csv(val_data_path, index_col = 0, parse_dates = True)

    return train_df, val_df

def test_dataset_loader():
    """
    Load test datasets for final model evaluation.

    Returns:
        test_df (DataFrame): Test dataset.
    """

    test_data_path = Path(__file__).resolve().parents[1] / "data" / "processed" / "wastewater_tank1_processed_test.csv"

    test_df = pd.read_csv(test_data_path, index_col = 0, parse_dates = True)

    return test_df
    

