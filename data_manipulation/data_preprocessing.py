import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import joblib

if __name__ == "__main__":
    df = pd.read_csv("../data/ENGINEERED_Melbourne_Housing_Market.csv")
    df["SaleDate"] = pd.to_datetime(df["SaleDate"])
    y = df["Price"]
    X = df.drop(columns=["Price"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    numerical_features = X.select_dtypes(include="number").columns
    low_card_cat_features = [column for column in X.select_dtypes(include="object").columns if
                             X[column].nunique() <= 10]
    high_card_cat_features = [column for column in X.select_dtypes(include="object").columns if
                              X[column].nunique() > 10]

    preprocessor = ColumnTransformer(transformers=[
        ("scaler", StandardScaler(), numerical_features),
        ("low_card_encoder", OneHotEncoder(handle_unknown="ignore"), low_card_cat_features),
        ("high_card_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         high_card_cat_features)
    ])
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    joblib.dump(preprocessor, "../utils/preprocessor.pkl")
    X_train_df = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())
    X_test_df = pd.DataFrame(X_test, columns=preprocessor.get_feature_names_out())
    y_train_df = pd.DataFrame(y_train)
    y_test_df = pd.DataFrame(y_test)
    X_train_df.to_csv("../data/split_data/X_train.csv", index=False)
    X_test_df.to_csv("../data/split_data/X_test.csv", index=False)
    y_train_df.to_csv("../data/split_data/y_train.csv", header=False, index=False)
    y_test_df.to_csv("../data/split_data/y_test.csv", header=False, index=False)
