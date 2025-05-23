import pandas as pd
import numpy as np
from utils.data_cleaning import reorder_df_columns


def remove_column(df_input: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df_input.copy()
    df.drop(columns=[col], inplace=True)
    return df


def separate_date(row: pd.Series) -> pd.Series:
    sale_year = row["SaleDate"].year
    sale_month = row["SaleDate"].month
    sale_day = row["SaleDate"].day
    sale_quarter = row["SaleDate"].quarter
    sale_day_of_week = row["SaleDate"].dayofweek
    return pd.Series({"SaleYear": sale_year, "SaleMonth": sale_month, "SaleDay": sale_day, "SaleQuarter": sale_quarter,
                      "SaleDayOfWeek": sale_day_of_week})


def separate_address(row: pd.Series) -> pd.Series:
    name, street_type = row["Address"].split("_")[-2:]
    return pd.Series({"StreetName": name, "StreetType": street_type})


def get_full_street_type(row: pd.Series) -> str:
    street_type_abbreviations = {"St": "Street", "Rd": "Road", "Av": "Avenue", "Ct": "Court", "Dr": "Drive",
                                 "Cr": "Crescent", "Gr": "Grove", "Pl": "Place", "Pde": "Parade", "Cl": "Close",
                                 "Wy": "Way", "La": "Lane", "Bvd": "Boulevard", "Tce": "Terrace", "Cct": "Circuit",
                                 "Hwy": "Highway", "Avenue": "Avenue", "Ri": "Rise", "Wk": "Walk", "Mw": "Meander Way",
                                 "Boulevard": "Boulevard", "Sq": "Square", "Parade": "Parade", "Esplanade": "Esplanade",
                                 "N": "North", "Qd": "Quay", "Cir": "Circle", "Vw": "View", "S": "South",
                                 "Crescent": "Crescent", "Prm": "Promenade", "Gdns": "Gardens", "W": "West",
                                 "Strand": "Strand", "Grove": "Grove", "Ridge": "Ridge", "Vs": "Views", "Ch": "Chase",
                                 "Fairway": "Fairway", "Righi": "Right", "E": "East", "Grn": "Green", "Wyn": "Way",
                                 "Gln": "Glen", "Esp": "Esplanade", "Bnd": "Bend", "Mews": "Mews", "Rdg": "Ridge",
                                 "Pky": "Parkway", "Gra": "Grange", "Rt": "Route", "Res": "Reserve", "Wky": "Way",
                                 "East": "East", "Lk": "Lake", "Nk": "Nook", "Gwy": "Gateway", "Mall": "Mall",
                                 "Highway": "Highway", "Ambl": "Ambleside", "Terrace": "Terrace", "Pt": "Point",
                                 "Parkway": "Parkway", "Street": "Street", "Corso": "Corso", "Outlook": "Outlook",
                                 "Media": "Media", "Hub": "Hub", "Crofts": "Crofts", "Victoria": "Victoria",
                                 "Nth": "North", "Athol": "Athol", "Nook": "Nook", "Rise": "Rise",
                                 "Greenway": "Greenway", "Views": "Views", "street": "Street", "Hl": "Hill",
                                 "Glade": "Glade", "Cove": "Cove", "Qy": "Quay", "Lairidge": "Lairidge",
                                 "Scala": "Scala", "Broadway": "Broadway", "Road": "Road", "Prst": "Prestwick",
                                 "Grand": "Grand", "Loop": "Loop", "Eyrie": "Eyrie", "Dell": "Dell", "Gve": "Grove",
                                 "Pkt": "Pocket", "Al": "Alley", "West": "West", "Hts": "Heights", "Aveue": "Avenue",
                                 "Summit": "Summit", "Ave": "Avenue", "Woodland": "Woodland", "Edg": "Edge",
                                 "Skyline": "Skyline", "Out": "Outlook", "Range": "Range", "Hth": "Heath",
                                 "Atrium": "Atrium", "Gables": "Gables", "Mears": "Mears", "App": "Approach",
                                 "Brk": "Brook", "Spur": "Spur", "Court": "Court", "Pass": "Pass", "Gld": "Gold",
                                 "Crse": "Course", "Ps": "Passage", "Entrance": "Entrance", "Heights": "Heights",
                                 "Boulevarde": "Boulevarde", "Circuit": "Circuit", "Parks": "Parks",
                                 "Ridgeway": "Ridgeway", "Panorama": "Panorama", "Briars": "Briars"}
    return street_type_abbreviations.get(row["StreetType"], "Street")


def calc_property_age(row: pd.Series) -> np.float64:
    return row["SaleDate"].year - row["YearBuilt"]


def calc_avg_room_size(row: pd.Series) -> np.float64:
    return row["BuildingArea"] / row["Rooms"]


def calc_building_to_land_ratio(row: pd.Series) -> np.float64:
    try:
        return row["BuildingArea"] / row["LandSize"]
    except ZeroDivisionError:
        return np.float64(1.0)


if __name__ == "__main__":
    dataset = pd.read_csv("../data/CLEANED_Melbourne_Housing_Market.csv")
    dataset["SaleDate"] = pd.to_datetime(dataset["SaleDate"])
    dataset[["SaleYear", "SaleMonth", "SaleDay", "SaleQuarter", "SaleDayOfWeek"]] = dataset.apply(separate_date, axis=1)
    dataset[["StreetName", "StreetType"]] = dataset.apply(separate_address, axis=1)
    dataset["StreetType"] = dataset.apply(get_full_street_type, axis=1)
    dataset = remove_column(dataset, "Address")
    dataset["PropertyAge"] = dataset.apply(calc_property_age, axis=1)
    dataset["AvgRoomSize"] = dataset.apply(calc_avg_room_size, axis=1)
    dataset["BuildingToLandRatio"] = dataset.apply(calc_building_to_land_ratio, axis=1)
    dataset = reorder_df_columns(dataset)
    dataset.to_csv("../data/ENGINEERED_Melbourne_Housing_Market.csv", index=False)
