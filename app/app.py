import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Food Delivery Time Predictor",
    page_icon="⏳",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "food_delivery_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(
        "Could not load the model. Please check that "
        "models/food_delivery_model.pkl exists."
    )
    st.exception(e)
    st.stop()

# -----------------------------
# App heading
# -----------------------------
st.title("⏳ Food Delivery Time Predictor")
st.write(
    "Predict estimated food delivery time using "
    "delivery distance, traffic, weather, and other factors."
)

st.info("Enter the delivery details below to get a prediction.")

# Input form
with st.form("delivery_form"):

    st.subheader("📍 Delivery Details")

    distance_km = st.number_input(
        "Delivery distance (km)",
        min_value=0.1,
        max_value=100.0,
        value=5.0,
        step=0.5
    )

    road_traffic_density = st.selectbox(
        "Road traffic density",
        ["Low", "Medium", "High", "Jam"]
    )

    weather_conditions = st.selectbox(
        "Weather conditions",
        ["Sunny", "Cloudy", "Windy", "Fog", "Stormy"]
    )

    city = st.selectbox(
        "City type",
        ["Urban", "Metropolitian", "Semi-Urban"]
    )

    type_of_vehicle = st.selectbox(
        "Vehicle type",
        ["motorcycle", "scooter", "electric_scooter", "bicycle"]
    )

    multiple_deliveries = st.selectbox(
        "Number of deliveries",
        [0, 1, 2, 3]
    )

    st.subheader("🛵 Delivery Partner Details")

    delivery_person_age = st.number_input(
        "Delivery partner age",
        min_value=18,
        max_value=60,
        value=28,
        step=1
    )

    delivery_person_ratings = st.slider(
        "Delivery partner rating",
        min_value=1.0,
        max_value=5.0,
        value=4.5,
        step=0.1
    )

    vehicle_condition = st.selectbox(
        "Vehicle condition",
        [0, 1, 2]
    )

    type_of_order = st.selectbox(
        "Type of order",
        ["Snack", "Meal", "Drinks", "Buffet"]
    )

    submitted = st.form_submit_button(
        "Predict Delivery Time 🚀",
        use_container_width=True
    )

# Prediction

if submitted:

    input_data = pd.DataFrame([{
        "distance_km": distance_km,
        "Road_traffic_density": road_traffic_density,
        "Weather_conditions": weather_conditions,
        "City": city,
        "Type_of_vehicle": type_of_vehicle,
        "multiple_deliveries": multiple_deliveries,
        "Delivery_person_Age": delivery_person_age,
        "Delivery_person_Ratings": delivery_person_ratings,
        "Vehicle_condition": vehicle_condition,
        "Type_of_order": type_of_order
    }])

    try:
        prediction = model.predict(input_data)[0]
        prediction = max(0, prediction)

        st.success("Prediction completed successfully!")

        st.metric(
            label="Estimated Delivery Time",
            value=f"{prediction:.1f} minutes"
        )

        st.caption(
            "This is a machine-learning estimate, "
            "not a guaranteed delivery time."
        )

    except Exception as e:
        st.error("Prediction failed. Please check the model inputs.")
        st.exception(e)

# Footer

st.divider()
st.caption("Built with Python, Scikit-learn, Pandas and Streamlit.")

