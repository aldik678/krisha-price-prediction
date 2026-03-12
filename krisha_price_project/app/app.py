from pathlib import Path
import json
import sys
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

MODEL_PATH = ROOT / "artifacts" / "model_pipeline.pkl"
METADATA_PATH = ROOT / "artifacts" / "app_metadata.json"

model = joblib.load(MODEL_PATH)
metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

st.set_page_config(page_title="Astana Apartment Price Predictor", layout="centered")

st.title("Astana Apartment Price Predictor")

with st.expander("About this model"):
    st.write(
        "The app uses the same saved preprocessing-and-model pipeline as the notebook. "
        "That avoids feature mismatch between training and deployment."
    )

district = st.selectbox("District", metadata["district_options"])
tip_doma = st.selectbox("Building type", metadata["tip_doma_options"])

jk_placeholder = ", ".join(metadata["example_jk_values"][1:6])
jk = st.text_input("Residential complex (JK)", placeholder=f"Examples: {jk_placeholder}")

year = st.number_input("Year built", min_value=1950, max_value=2035, value=2022)
floor = st.number_input("Floor", min_value=0, max_value=100, value=5)
area = st.number_input("Area (m²)", min_value=10.0, max_value=500.0, value=50.0)

toilet = st.selectbox("Bathroom type", metadata["toilet_options"])
parking = st.selectbox("Parking", metadata["parking_options"])
status = st.selectbox("Renovation status", metadata["status_options"])
num_rooms = st.number_input("Number of rooms", min_value=1, max_value=12, value=2)

if st.button("Predict price"):
    user_input = pd.DataFrame(
        {
            "district": [district],
            "tip_doma": [tip_doma],
            "jk": [jk if jk.strip() else "No data"],
            "year": [year],
            "floor": [floor],
            "area": [area],
            "toilet": [toilet],
            "parking": [parking],
            "status": [status],
            "num_rooms": [num_rooms],
        }
    )

    prediction = float(model.predict(user_input)[0])

    st.success(f"Estimated price: {prediction:,.0f} ₸")
    st.write(f"Approx. {prediction / 1_000_000:.1f} million ₸")