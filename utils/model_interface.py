import pandas as pd
import utils.feature_engineering as fe
import joblib

preprocessor = None
model = None


def load_preprocessor_and_model(prefix: str = ""):
    global preprocessor
    global model
    preprocessor = joblib.load(f"{prefix}raw/preprocessor.pkl")
    model = joblib.load(f"{prefix}raw/GradientBoostingRegressor.pkl")


def predict_from_input(user_input: pd.Series) -> str:
    separated_date = fe.separate_date(user_input)
    for metric, value in separated_date.items():
        user_input[metric] = value
    user_input["PropertyAge"] = fe.calc_property_age(user_input)
    user_input["AvgRoomSize"] = fe.calc_avg_room_size(user_input)
    user_input["BuildingToLandRatio"] = fe.calc_building_to_land_ratio(user_input)
    user_input_df = user_input.to_frame().T
    user_input = preprocessor.transform(user_input_df)
    feature_names = preprocessor.get_feature_names_out()
    user_input_df = pd.DataFrame(user_input, columns=feature_names)
    prediction = model.predict(user_input_df)
    return prediction[0]


if __name__ == "__main__":
    first_row = pd.read_csv("../data/CLEANED_Melbourne_Housing_Market.csv").iloc[0]
    first_row = first_row.drop(["Address", "SaleMethod", "UnitType"])
    first_row["SaleDate"] = pd.to_datetime(first_row["SaleDate"])
    real_price = first_row["Price"]
    first_row = first_row.drop("Price")
    load_preprocessor_and_model("../")
    predict = predict_from_input(first_row)
    print("Row:")
    print(first_row)
    print(f"was predicted to be: {predict}")
    print(f"It is in fact {real_price}.")
