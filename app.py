# ==========================================
# SMART ELECTRICITY CONSUMPTION PREDICTION
# STREAMLIT APPLICATION
# ==========================================

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from pathlib import Path


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(

    page_title="Smart Energy Predictor",

    page_icon="⚡",

    layout="wide"

)


# ==========================================
# 2. PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

METRICS_PATH = BASE_DIR / "models" / "metrics.json"

DATA_PATH = BASE_DIR / "energydata_complete.csv"


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.title("⚡ Smart Electricity Consumption Predictor")

st.markdown(
    """
    ### Machine Learning Based Energy Prediction System

    Predict household appliance energy consumption
    using environmental and time-related information.
    """
)

st.divider()


# ==========================================
# 4. CHECK MODEL FILES
# ==========================================

if not MODEL_PATH.exists() or not METRICS_PATH.exists():

    st.error(
        "Trained model not found! "
        "Please execute python train.py first."
    )

    st.stop()


# ==========================================
# 5. LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


model = load_model()


with open(METRICS_PATH, "r") as file:

    metrics = json.load(file)


defaults = metrics["defaults"]


# ==========================================
# 6. SIDEBAR
# ==========================================

st.sidebar.title("⚡ Project Information")

st.sidebar.write(
    "**Project:** Electricity Consumption Prediction"
)

st.sidebar.write(
    "**Technique:** Machine Learning"
)

st.sidebar.write(
    "**Task:** Regression"
)

st.sidebar.write(
    "**Dataset:** UCI Appliances Energy"
)

st.sidebar.write(
    "**Best Model:**"
)

st.sidebar.success(metrics["best_model"])


st.sidebar.divider()

st.sidebar.write("Developed using Python and Streamlit")


# ==========================================
# 7. DASHBOARD METRICS
# ==========================================

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)


col1.metric(
    "Total Records",
    f"{metrics['dataset_records']:,}"
)

col2.metric(
    "Training Records",
    f"{metrics['training_records']:,}"
)

col3.metric(
    "Testing Records",
    f"{metrics['testing_records']:,}"
)


st.divider()


# ==========================================
# 8. USER INPUT SECTION
# ==========================================

st.header("🏠 Enter Environmental Conditions")

st.write(
    "Enter the environmental conditions to predict "
    "appliance energy consumption."
)


with st.form("prediction_form"):

    col1, col2 = st.columns(2)


    # --------------------------------------
    # INDOOR CONDITIONS
    # --------------------------------------

    with col1:

        st.subheader("🌡️ Indoor Conditions")


        lights = st.number_input(

            "Lights Consumption (Wh)",

            min_value=0.0,

            value=float(defaults["lights"])

        )


        t1 = st.number_input(

            "Kitchen Temperature (°C)",

            value=float(defaults["T1"])

        )


        rh1 = st.number_input(

            "Kitchen Humidity (%)",

            min_value=0.0,

            max_value=100.0,

            value=float(defaults["RH_1"])

        )


        t2 = st.number_input(

            "Living Room Temperature (°C)",

            value=float(defaults["T2"])

        )


        rh2 = st.number_input(

            "Living Room Humidity (%)",

            min_value=0.0,

            max_value=100.0,

            value=float(defaults["RH_2"])

        )


    # --------------------------------------
    # OUTDOOR CONDITIONS
    # --------------------------------------

    with col2:

        st.subheader("🌤️ Outdoor Conditions")


        t_out = st.number_input(

            "Outside Temperature (°C)",

            value=float(defaults["T_out"])

        )


        rh_out = st.number_input(

            "Outside Humidity (%)",

            min_value=0.0,

            max_value=100.0,

            value=float(defaults["RH_out"])

        )


        windspeed = st.number_input(

            "Wind Speed (m/s)",

            min_value=0.0,

            value=float(defaults["Windspeed"])

        )


        hour = st.slider(

            "Select Hour",

            min_value=0,

            max_value=23,

            value=19

        )


        day = st.selectbox(

            "Select Day",

            options=list(range(7)),

            format_func=lambda x: [

                "Monday",

                "Tuesday",

                "Wednesday",

                "Thursday",

                "Friday",

                "Saturday",

                "Sunday"

            ][x]

        )


    st.divider()


    tariff = st.number_input(

        "Electricity Tariff (₹ per kWh)",

        min_value=0.0,

        value=8.0

    )


    submit = st.form_submit_button(

        "⚡ Predict Electricity Consumption",

        use_container_width=True

    )


# ==========================================
# 9. PREDICTION
# ==========================================

if submit:

    # Convert time into numerical features

    hour_sin = np.sin(

        2 * np.pi * hour / 24

    )

    hour_cos = np.cos(

        2 * np.pi * hour / 24

    )


    # Prepare input data

    input_data = pd.DataFrame([{

        "lights": lights,

        "T1": t1,

        "RH_1": rh1,

        "T2": t2,

        "RH_2": rh2,

        "T_out": t_out,

        "RH_out": rh_out,

        "Windspeed": windspeed,

        "hour_sin": hour_sin,

        "hour_cos": hour_cos,

        "day_of_week": day

    }])


    # Generate prediction

    prediction = float(

        model.predict(input_data)[0]

    )

    prediction = max(0, prediction)


    # Convert Wh into kWh

    energy_kwh = prediction / 1000


    # Calculate estimated cost

    estimated_cost = energy_kwh * tariff


    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    st.divider()

    st.header("⚡ Prediction Results")


    col1, col2, col3 = st.columns(3)


    col1.metric(

        "Predicted Energy",

        f"{prediction:.2f} Wh"

    )


    col2.metric(

        "Energy Consumption",

        f"{energy_kwh:.4f} kWh"

    )


    col3.metric(

        "Estimated Electricity Cost",

        f"₹{estimated_cost:.4f}"

    )


    st.success(

        "Energy consumption prediction completed successfully!"

    )


    st.info(

        "The predicted value represents appliance energy "
        "consumption for a 10-minute interval."
    )


# ==========================================
# 10. MODEL PERFORMANCE
# ==========================================

st.divider()

st.header("📈 Machine Learning Model Performance")


results = pd.DataFrame(

    metrics["results"]

).T


results.index.name = "Algorithm"


st.subheader("Algorithm Comparison")


st.dataframe(

    results,

    use_container_width=True

)


# ==========================================
# 11. PERFORMANCE CHART
# ==========================================

st.subheader("Model Error Comparison")


st.bar_chart(

    results[["MAE", "RMSE"]]

)


# ==========================================
# 12. ACTUAL VS PREDICTED
# ==========================================

st.divider()

st.header("📊 Actual vs Predicted Consumption")


comparison = pd.DataFrame({

    "Actual Consumption": metrics["test_actual"],

    "Predicted Consumption": metrics["test_predicted"]

})


st.line_chart(comparison)


# ==========================================
# 13. HISTORICAL DATA ANALYSIS
# ==========================================

st.divider()

st.header("📉 Historical Energy Consumption")


if DATA_PATH.exists():

    historical_df = pd.read_csv(DATA_PATH)


    historical_df["date"] = pd.to_datetime(

        historical_df["date"],

        format="%Y-%m-%d %H:%M:%S"

    )


    historical_df = historical_df.sort_values("date")


    historical_df["Date"] = (

        historical_df["date"].dt.date

    )


    daily_consumption = (

        historical_df.groupby("Date")["Appliances"]

        .mean()

    )


    st.subheader("Average Appliance Energy Consumption")


    st.line_chart(daily_consumption)


    st.caption(

        "Historical average consumption per 10-minute observation, "
        "grouped by date."

    )


else:

    st.warning("Historical dataset not found.")


# ==========================================
# 14. FOOTER
# ==========================================

st.divider()

st.caption(

    "Smart Electricity Consumption Prediction System | "
    "Developed using Python, Scikit-learn and Streamlit."

)

st.caption(

    "Predictions are estimates based on historical dataset patterns. "
    "They are not guaranteed measurements of actual electricity usage."

)