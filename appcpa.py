import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Load the dataset
df = pd.read_csv('CarPrice_Assignment.csv')

# Drop unnecessary columns
X = df.drop(['car_ID', 'CarName', 'price'], axis=1)

# Preprocess categorical columns
categorical_cols = [
    'fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation',
    'enginetype', 'cylindernumber', 'fuelsystem'
]
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
rf = joblib.load('random_forest_model.joblib')

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
# Load the dataset using a relative path
df = pd.read_csv('CarPrice_Assignment.csv')

# Drop unnecessary columns
X = df.drop(['car_ID', 'CarName', 'price'], axis=1)
y = df['price']

# Preprocessing
X['doornumber'] = X['doornumber'].replace({'two': 2, 'four': 4}).astype(int)

# Load the trained model
rf = joblib.load('random_forest_model.pkl')

# Streamlit App Interface
st.title("Car Price Prediction")

# Input fields
fueltype = st.selectbox("Fuel Type", X['fueltype'].unique())
aspiration = st.selectbox("Aspiration", X['aspiration'].unique())
doornumber = st.selectbox("Number of Doors", X['doornumber'].unique())
carbody = st.selectbox("Car Body", X['carbody'].unique())
drivewheel = st.selectbox("Drive Wheel", X['drivewheel'].unique())
enginelocation = st.selectbox("Engine Location", X['enginelocation'].unique())
wheelbase = st.number_input("Wheelbase", min_value=X['wheelbase'].min(), max_value=X['wheelbase'].max())
carlength = st.number_input("Car Length", min_value=X['carlength'].min(), max_value=X['carlength'].max())
carwidth = st.number_input("Car Width", min_value=X['carwidth'].min(), max_value=X['carwidth'].max())
carheight = st.number_input("Car Height", min_value=X['carheight'].min(), max_value=X['carheight'].max())
curbweight = st.number_input("Curb Weight", min_value=X['curbweight'].min(), max_value=X['curbweight'].max())
enginetype = st.selectbox("Engine Type", X['enginetype'].unique())
cylindernumber = st.selectbox("Cylinder Number", X['cylindernumber'].unique())
enginesize = st.number_input("Engine Size", min_value=X['enginesize'].min(), max_value=X['enginesize'].max())
fuelsystem = st.selectbox("Fuel System", X['fuelsystem'].unique())
boreratio = st.number_input("Bore Ratio", min_value=X['boreratio'].min(), max_value=X['boreratio'].max())
stroke = st.number_input("Stroke", min_value=X['stroke'].min(), max_value=X['stroke'].max())
compressionratio = st.number_input("Compression Ratio", min_value=X['compressionratio'].min(), max_value=X['compressionratio'].max())
horsepower = st.number_input("Horsepower", min_value=X['horsepower'].min(), max_value=X['horsepower'].max())
peakrpm = st.number_input("Peak RPM", min_value=X['peakrpm'].min(), max_value=X['peakrpm'].max())
citympg = st.number_input("City MPG", min_value=X['citympg'].min(), max_value=X['citympg'].max())
highwaympg = st.number_input("Highway MPG", min_value=X['highwaympg'].min(), max_value=X['highwaympg'].max())

# Predict button
if st.button("Predict"):
    input_data = pd.DataFrame([[fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation,
                                wheelbase, carlength, carwidth, carheight, curbweight, enginetype,
                                cylindernumber, enginesize, fuelsystem, boreratio, stroke, compressionratio,
                                horsepower, peakrpm, citympg, highwaympg]],
                              columns=X.columns)
    
    # Ensure input data has the same feature names as the training data
    prediction = rf.predict(input_data)
    st.success(f"Predicted Car Price: ${prediction[0]:.2f}")
