import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score, \
    mean_absolute_percentage_error
import joblib


def load_split_datasets() -> tuple:
    X_train = pd.read_csv("../data/split_data/X_train.csv")
    X_test = pd.read_csv("../data/split_data/X_test.csv")
    y_train = pd.read_csv("../data/split_data/y_train.csv", header=None).to_numpy().ravel()
    y_test = pd.read_csv("../data/split_data/y_test.csv", header=None).to_numpy().ravel()
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.array, y_test: np.array, model: BaseEstimator,
                evaluation: pd.DataFrame = None, save: bool = False) -> pd.DataFrame:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_name = model.__class__.__name__
    if evaluation is not None:
        evaluation.at[model_name, "mae"] = mean_absolute_error(y_test, y_pred)
        evaluation.at[model_name, "mse"] = mean_squared_error(y_test, y_pred)
        evaluation.at[model_name, "rmse"] = root_mean_squared_error(y_test, y_pred)
        evaluation.at[model_name, "r2"] = r2_score(y_test, y_pred)
        evaluation.at[model_name, "mape"] = mean_absolute_percentage_error(y_test, y_pred)
    if save:
        joblib.dump(model, f"../raw/{model_name}")
    return evaluation


if __name__ == "__main__":
    pass
