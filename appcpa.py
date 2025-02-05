import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('C:/VS Code/streamlit/Car Price Prediction/CarPrice_Assignment.csv')

# Drop unnecessary columns
X = df.drop(['car_ID', 'CarName', 'price'], axis=1)

# Preprocess categorical columns
categorical_cols = ['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',
                    'enginetype', 'cylindernumber', 'fuelsystem']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Handle 'doornumber' separately
X['doornumber'] = X['doornumber'].replace({'two': 2, 'four': 4})

# Save encoders (optional, if needed later)
joblib.dump(label_encoders, 'label_encoders.joblib')

# Load the trained model
rf = joblib.load('C:/VS Code/streamlit/Car Price Prediction/random_forest_model.joblib')

# Load the label encoders
label_encoders = joblib.load('label_encoders.joblib')

# Define the prediction function
def predict_car_price(
    fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation,
    wheelbase, carlength, carwidth, carheight, curbweight, enginetype,
    cylindernumber, enginesize, fuelsystem, boreratio, stroke, compressionratio,
    horsepower, peakrpm, citympg, highwaympg
):
    # Map categorical inputs using precomputed encodings
    input_data = np.array([
        label_encoders['fueltype'].transform([fueltype])[0],
        label_encoders['aspiration'].transform([aspiration])[0],
        doornumber,
        label_encoders['carbody'].transform([carbody])[0],
        label_encoders['drivewheel'].transform([drivewheel])[0],
        label_encoders['enginelocation'].transform([enginelocation])[0],
        wheelbase, carlength, carwidth, carheight, curbweight,
        label_encoders['enginetype'].transform([enginetype])[0],
        label_encoders['cylindernumber'].transform([cylindernumber])[0],
        enginesize,
        label_encoders['fuelsystem'].transform([fuelsystem])[0],
        boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg
    ]).reshape(1, -1)
    
    # Predict using the trained model
    prediction = rf.predict(input_data)
    return f"Predicted Car Price: ${prediction[0]:.2f}"

# Streamlit App Interface
st.title("Car Price Prediction")
st.write("Enter the car's features to predict its price.")

# Input fields
fueltype = st.selectbox("Fuel Type", options=df['fueltype'].unique())
aspiration = st.selectbox("Aspiration", options=df['aspiration'].unique())
doornumber = st.selectbox("Number of Doors", options=[2, 4])
carbody = st.selectbox("Car Body", options=df['carbody'].unique())
drivewheel = st.selectbox("Drive Wheel", options=df['drivewheel'].unique())
enginelocation = st.selectbox("Engine Location", options=df['enginelocation'].unique())
wheelbase = st.slider("Wheelbase", min_value=float(df['wheelbase'].min()), max_value=float(df['wheelbase'].max()))
carlength = st.slider("Car Length", min_value=float(df['carlength'].min()), max_value=float(df['carlength'].max()))
carwidth = st.slider("Car Width", min_value=float(df['carwidth'].min()), max_value=float(df['carwidth'].max()))
carheight = st.slider("Car Height", min_value=float(df['carheight'].min()), max_value=float(df['carheight'].max()))
curbweight = st.slider("Curb Weight", min_value=int(df['curbweight'].min()), max_value=int(df['curbweight'].max()))
enginetype = st.selectbox("Engine Type", options=df['enginetype'].unique())
cylindernumber = st.selectbox("Number of Cylinders", options=df['cylindernumber'].unique())
enginesize = st.slider("Engine Size", min_value=int(df['enginesize'].min()), max_value=int(df['enginesize'].max()))
fuelsystem = st.selectbox("Fuel System", options=df['fuelsystem'].unique())
boreratio = st.slider("Bore Ratio", min_value=float(df['boreratio'].min()), max_value=float(df['boreratio'].max()))
stroke = st.slider("Stroke", min_value=float(df['stroke'].min()), max_value=float(df['stroke'].max()))
compressionratio = st.slider("Compression Ratio", min_value=float(df['compressionratio'].min()), max_value=float(df['compressionratio'].max()))
horsepower = st.slider("Horsepower", min_value=int(df['horsepower'].min()), max_value=int(df['horsepower'].max()))
peakrpm = st.slider("Peak RPM", min_value=int(df['peakrpm'].min()), max_value=int(df['peakrpm'].max()))
citympg = st.slider("City MPG", min_value=int(df['citympg'].min()), max_value=int(df['citympg'].max()))
highwaympg = st.slider("Highway MPG", min_value=int(df['highwaympg'].min()), max_value=int(df['highwaympg'].max()))

# Predict button
if st.button("Predict"):
    result = predict_car_price(
        fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation,
        wheelbase, carlength, carwidth, carheight, curbweight, enginetype,
        cylindernumber, enginesize, fuelsystem, boreratio, stroke, compressionratio,
        horsepower, peakrpm, citympg, highwaympg
    )
    st.success(result)