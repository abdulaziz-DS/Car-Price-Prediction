import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
import joblib

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
