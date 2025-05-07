import pandas as pd
from data_manipulation.feature_engineering import remove_column
import math
import matplotlib.pyplot as plt


def remove_outliers(df_input: pd.DataFrame, method="IQR") -> pd.DataFrame:
    df = df_input.copy()
    for col in df.select_dtypes(include="number").columns:
        outliers = get_outliers(df, col, method)
        df = df[~df.index.isin(outliers.index)]
    return df


def safe_round(val: float) -> float:
    try:
        return round(val, 2)
    except TypeError:
        return val


def get_outliers(df_input: pd.DataFrame, col: str, method: str = "iqr") -> pd.DataFrame:
    df = df_input.copy()
    outliers = pd.DataFrame()
    if method == "iqr":
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].copy()
    elif method == "zscore":
        zscore = (df[col] - df[col].mean()) / df[col].std()
        outliers = df[(zscore < -3) | (zscore > 3)].copy()
    return outliers


def get_custom_description(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    description = df.describe(include="all").T
    description["MajorityPercentage"] = description["freq"] / description["count"]
    description["Range"] = description["max"] - description["min"]
    description["IQR"] = description["75%"] - description["25%"]
    description["Skewness"] = df.select_dtypes(include="number").skew()
    description["Kurtosis"] = df.select_dtypes(include="number").kurtosis()
    description["IQROutliers"] = df.select_dtypes(include="number").apply(
        lambda col: len(get_outliers(df, col.name, method="iqr")))
    description["ZScoreOutliers"] = df.select_dtypes(include="number").apply(
        lambda col: len(get_outliers(df, col.name, method="zscore")))
    description = remove_column(description, "count")
    description = remove_column(description, "freq")
    description = remove_column(description, "25%")
    description = remove_column(description, "50%")
    description = remove_column(description, "75%")
    description = description.rename(
        columns={"unique": "Cardinality", "top": "Majority", "mean": "Mean", "min": "Min", "max": "Max", "std": "STD"})
    description = description.map(safe_round)
    description = description[
        ["Cardinality", "Majority", "MajorityPercentage", "Min", "Max", "Range", "Skewness", "IQR", "IQROutliers",
         "Kurtosis", "Mean", "STD", "ZScoreOutliers"]]
    return description.T


def top_n_filter(df_input: pd.DataFrame, col: str, n: int = 5) -> pd.DataFrame:
    df = df_input.copy()
    column_value_counts = df[col].value_counts()
    top_n_values = column_value_counts.head(n).index
    df[col] = df[col].apply(lambda value: value if value in top_n_values else "Other")
    return df


def plot_these(*funcs, **kwargs) -> None:
    _, axes = plt.subplots((len(funcs) + 1) // 2, 2, figsize=(10, 2 * len(funcs)))
    axes = axes.ravel()
    for index, plot in enumerate(funcs):
        plot(**kwargs, ax=axes[index])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = pd.read_csv("../data/ENGINEERED_Melbourne_Housing_Market.csv")
    dataset["SaleDate"] = pd.to_datetime(dataset["SaleDate"])
    dataset = remove_column(dataset, "StreetName")
    dataset = remove_column(dataset, "SaleMethod")
    dataset = remove_column(dataset, "StreetType")
    dataset = remove_column(dataset, "UnitType")
    dataset = top_n_filter(dataset, "RealEstateAgent", 32)
    dataset = top_n_filter(dataset, "Suburb", 32)
    dataset = remove_outliers(dataset, "zscore")
