import pandas as pd
from sklearn.impute import SimpleImputer


def correct_column_names(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    df.rename(columns={"Type": "UnitType", "Method": "SaleMethod", "SellerG": "RealEstateAgent", "Date": "SaleDate",
                       "Distance": "DistanceToCBD", "Bedroom2": "Bedrooms", "Bathroom": "Bathrooms", "Car": "CarSpots",
                       "Landsize": "LandSize", "Lattitude": "Latitude", "Longtitude": "Longitude",
                       "Regionname": "RegionName", "Propertycount": "NeighbouringProperties"}, inplace=True)
    return df


def convert_floats_to_ints(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    should_be_float_columns = ["DistanceToCBD", "BuildingArea", "Latitude", "Longitude"]
    should_be_int_columns = [column for column in df.select_dtypes(include="number").columns if
                             column not in should_be_float_columns]
    for column in should_be_int_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    return df


def format_df_cells(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()

    def format_unit_type_cells(input_df_input: pd.DataFrame) -> pd.DataFrame:
        input_df = input_df_input
        unit_type_abbreviations = {"h": "House", "u": "Duplex", "t": "Town House"}
        input_df["UnitType"] = input_df["UnitType"].replace(unit_type_abbreviations)
        return input_df

    def format_sale_method_cells(input_df_input: pd.DataFrame) -> pd.DataFrame:
        input_df = input_df_input
        sale_method_abbreviations = {"S": "Sold", "SP": "Sold Prior", "PI": "Passed In", "VB": "Vendor Bid",
                                     "SN": "Sold Not Disclosed", "PN": "Sold Prior Not Disclosed",
                                     "SA": "Sold After Auction", "W": "Withdrawn Prior to Auction",
                                     "SS": "Sold After Auction Not Disclosed"}
        input_df["SaleMethod"] = input_df["SaleMethod"].replace(sale_method_abbreviations)
        return input_df

    def format_sale_date_cells(input_df_input: pd.DataFrame) -> pd.DataFrame:
        input_df = input_df_input
        input_df["SaleDate"] = pd.to_datetime(input_df["SaleDate"], format="mixed")
        return input_df

    def format_council_area_cells(input_df_input: pd.DataFrame) -> pd.DataFrame:
        input_df = input_df_input
        input_df["CouncilArea"] = input_df["CouncilArea"].str.replace(" Council", "")
        return input_df

    def replace_non_alpha_num_chars(input_df_input: pd.DataFrame) -> pd.DataFrame:
        input_df = input_df_input.copy()

        def get_non_alpha_num_chars(input_input_df_input: pd.DataFrame) -> set:
            input_input_df = input_input_df_input.copy()
            text_data = input_input_df.select_dtypes(include="object").astype(str).values.ravel()
            return {char for cell in text_data for char in cell if not char.isalnum()}

        non_alpha_num_chars = get_non_alpha_num_chars(input_df)
        for column in input_df.select_dtypes(include="object").columns:
            input_df[column] = input_df[column].str.strip().str.title()
            for non_alpha_num_char in non_alpha_num_chars:
                input_df[column] = input_df[column].str.replace(non_alpha_num_char, "_")
            input_df[column] = input_df[column].str.strip("_")
        return input_df

    df = format_unit_type_cells(df)
    df = format_sale_method_cells(df)
    df = format_sale_date_cells(df)
    df = format_council_area_cells(df)
    df = replace_non_alpha_num_chars(df)
    return df


def reorder_df_columns(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    df = df[sorted(df.select_dtypes(include="object").columns.tolist()) + sorted(
        df.select_dtypes(exclude="object").columns.tolist())]
    columns = ["SaleDate"] + [column for column in df.columns if column not in ["SaleDate", "Price"]] + ["Price"]
    df = df[columns]
    return df


def estimate_nulls(df_input: pd.DataFrame, remove_price=False) -> pd.DataFrame:
    df = df_input.copy()
    if remove_price:
        df.dropna(subset=["Price"], inplace=True)
    imputer = SimpleImputer(strategy="median")
    df["SaleDate"] = df["SaleDate"].astype("int64")
    df[df.select_dtypes(include="number").columns] = imputer.fit_transform(
        df[df.select_dtypes(include="number").columns])
    df["SaleDate"] = pd.to_datetime(df["SaleDate"])
    imputer = SimpleImputer(strategy="most_frequent")
    df[df.select_dtypes(include="object").columns] = imputer.fit_transform(
        df[df.select_dtypes(include="object").columns])
    return df


def drop_duplicates(df_input: pd.DataFrame) -> pd.DataFrame:
    df = df_input.copy()
    df.drop_duplicates(inplace=True)
    return df


if __name__ == "__main__":
    dataset = pd.read_csv("../data/Melbourne_Housing_Market.csv")
    dataset = correct_column_names(dataset)
    dataset = convert_floats_to_ints(dataset)
    dataset = format_df_cells(dataset)
    dataset = reorder_df_columns(dataset)
    dataset = estimate_nulls(dataset, remove_price=True)
    dataset = drop_duplicates(dataset)
    dataset.to_csv("../data/CLEANED_Melbourne_Housing_Market.csv", index=False)
