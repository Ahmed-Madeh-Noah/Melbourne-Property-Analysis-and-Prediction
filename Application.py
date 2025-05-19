import pandas as pd
import streamlit as st
import datetime as dt
from streamlit_folium import st_folium
import folium
from utils.model_interface import load_preprocessor_and_model, predict_from_input

load_preprocessor_and_model()

df = pd.read_csv("data/ANALYSED_Melbourne_Housing_Market.csv")

st.title("Melbourne Property Price Predictor")

lat_min, lat_max = df["Latitude"].min(), df["Latitude"].max()
lng_min, lng_max = df["Longitude"].min(), df["Longitude"].max()
lat_cen = ((lat_max - lat_min) / 2) + lat_min
lng_cen = ((lng_max - lng_min) / 2) + lng_min
if "clicked_point" not in st.session_state:
    st.session_state.clicked_point = None
m = folium.Map(location=[lat_cen, lng_cen], zoom_start=10)
m.add_child(folium.LatLngPopup())
map_data = st_folium(m, width=800, height=400)
if map_data and map_data.get("last_clicked"):
    st.session_state.clicked_point = map_data["last_clicked"]
if st.session_state.clicked_point:
    lat = st.session_state.clicked_point["lat"]
    lng = st.session_state.clicked_point["lng"]
    if lat < lat_min or lat > lat_max or lng < lng_min or lng > lng_max:
        st.error("Unsupported territory")
        lat, lng = lat_cen, lng_cen
    else:
        st.success(f"Latitude: {lat}, Longitude: {lng}")
else:
    st.write("Click on the map to get unit coordinates")
    lat, lng = lat_cen, lng_cen

with st.form("model_input_form"):
    sale_date = st.date_input("Pick a Sale Date:", value=dt.date.today())

    year_built = st.date_input("When was it built?", value=dt.date.today() - dt.timedelta(days=20 * 365)).year

    region_names = sorted(df["RegionName"].unique())
    region_name = st.radio("To which Region does it belong?", region_names, horizontal=True)

    suburbs = sorted(df["Suburb"].unique())
    suburb = st.selectbox("To which Suburb does it belong?", suburbs)

    council_areas = sorted(df["CouncilArea"].unique())
    council_area = st.selectbox("To which Suburb does it belong?", council_areas)

    min_distance_to_cbd, max_distance_to_cbd = df["DistanceToCBD"].min(), df["DistanceToCBD"].max()
    distance_to_cbd = st.slider("Distance to City Center (Kilometers):", min_value=min_distance_to_cbd,
                                max_value=max_distance_to_cbd)

    postcode = st.number_input("Enter the unit's Postcode:", min_value=0, max_value=10000, value=1000)

    neighbouring_properties = st.number_input("Enter the number of Neighbouring Properties:", min_value=0,
                                              max_value=20000, value=500)

    real_estate_agents = sorted(df["RealEstateAgent"].unique())
    real_estate_agent = st.selectbox("Which Real Estate Agent is offering?", real_estate_agents)

    land_size = st.number_input("What is the size of the Land (Square Meters)?", min_value=0, max_value=10000,
                                value=500)

    building_area = st.number_input("What is the size of the Unit (Square Meters)?", min_value=0,
                                    max_value=land_size, value=250)

    bedrooms = sorted(df["Bedrooms"].unique().astype(int))
    n_bedrooms = st.radio("How many Bedrooms in the unit?", bedrooms, horizontal=True)

    rooms = sorted(df["Rooms"].unique().astype(int))
    n_rooms = st.radio("How many other rooms are there?", rooms, horizontal=True)

    bathrooms = sorted(df["Bathrooms"].unique().astype(int))
    n_bathrooms = st.radio("How many Bathrooms in the unit?", bathrooms, horizontal=True)

    car_spots = sorted(df["CarSpots"].unique().astype(int))
    n_car_spots = st.radio("How many Car Spots in the unit?", car_spots, horizontal=True)

    submitted = st.form_submit_button("Predict")

    if submitted:
        user_input = pd.Series({
            "Latitude": lat,
            "Longitude": lng,
            "SaleDate": pd.to_datetime(sale_date),
            "YearBuilt": year_built,
            "RegionName": region_name,
            "Suburb": suburb,
            "CouncilArea": council_area,
            "DistanceToCBD": distance_to_cbd,
            "Postcode": postcode,
            "NeighbouringProperties": neighbouring_properties,
            "RealEstateAgent": real_estate_agent,
            "LandSize": land_size,
            "BuildingArea": building_area,
            "Rooms": n_rooms,
            "Bedrooms": n_bedrooms,
            "Bathrooms": n_bathrooms,
            "CarSpots": n_car_spots
        })
        pred = predict_from_input(user_input)
        st.success(f"ML Model Prediction: ${pred:,.2f}")
