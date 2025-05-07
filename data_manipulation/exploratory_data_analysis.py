import pandas as pd
from data_manipulation.feature_engineering import remove_column


def remove_outliers(df: pd.DataFrame, method="IQR") -> pd.DataFrame:
    for col in df.select_dtypes(include="number").columns:
        outliers = get_outliers(df, col, method)
        df = df[~df.index.isin(outliers.index)]
    return df


def safe_round(val: float) -> float:
    try:
        return round(val, 2)
    except TypeError:
        return val


def get_outliers(df: pd.DataFrame, col: str, method: str = "iqr") -> pd.DataFrame:
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


def get_custom_description(df: pd.DataFrame) -> pd.DataFrame:
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


if __name__ == "__main__":
    dataset = pd.read_csv("../data/ENGINEERED_Melbourne_Housing_Market.csv")
    dataset = remove_column(dataset, "StreetName")
    dataset = remove_column(dataset, "SaleMethod")
    dataset = remove_column(dataset, "StreetType")
    dataset = remove_column(dataset, "UnitType")
    dataset = remove_outliers(dataset, "zcore")
    dataset = dataset[dataset["YearBuilt"] >= dataset["YearBuilt"].median() - (
                dataset["YearBuilt"].max() - dataset["YearBuilt"].median())]
