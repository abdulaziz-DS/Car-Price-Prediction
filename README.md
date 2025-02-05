# Car Price Prediction

This project focuses on predicting car prices using machine learning techniques. The dataset includes various car features such as fuel type, engine size, horsepower, and more. Multiple regression models were implemented, and a **Gradio** and **Streamlit**-based web application was deployed for real-time predictions.

## Features

- **Exploratory Data Analysis (EDA):**
  - Visualization and statistical analysis of dataset features.
  - Identification of outliers and feature distributions.

- **Data Preprocessing:**
  - Handling categorical and numerical data.
  - Label encoding for categorical variables.
  - Splitting the dataset into training and testing sets.

- **Machine Learning Models Implemented:**
  - **Linear Regression (Accuracy: 73%)**
  - **Support Vector Machine (SVM) (Accuracy: 79%)**
  - **Random Forest (Accuracy: 91%)**
  - **Decision Tree (Accuracy: 80%)**

- **Model Evaluation:**
  - Performance metrics such as R² score and mean squared error (MSE).
  - Comparison of different models to determine the most effective predictor.

- **Deployment with Gradio and Streamlit:**
  - Interactive web-based UIs for car price prediction.
  - Users can input car features and receive instant price predictions.

## Dataset

The dataset used contains various car features such as:
- **Fuel Type**
- **Aspiration**
- **Number of Doors**
- **Car Body Type**
- **Drive Wheel**
- **Engine Location**
- **Wheelbase**
- **Car Length**
- **Car Width**
- **Car Height**
- **Curb Weight**
- **Engine Type**
- **Number of Cylinders**
- **Engine Size**
- **Fuel System**
- **Bore Ratio**
- **Stroke**
- **Compression Ratio**
- **Horsepower**
- **Peak RPM**
- **City MPG**
- **Highway MPG**
- **Price (Target Variable)**

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have the dataset in CSV format and place it in the appropriate directory.

## Usage

### Running the Jupyter Notebook
1. Open `Car_Price_Prediction.ipynb` in Jupyter Notebook.
2. Run all cells to:
   - Perform data preprocessing and EDA.
   - Train and evaluate different regression models.
   - Compare accuracy metrics.

### Running the Gradio Web App
1. Start the Gradio interface:
   ```bash
   python app_gradio.py
   ```
2. Enter car features in the UI.
3. Receive an instant prediction of the car price.

### Running the Streamlit Web App
1. Start the Streamlit interface:
   ```bash
   streamlit run app_streamlit.py
   ```
2. Enter car features in the UI.
3. Receive an instant prediction of the car price.

## Results

- **Random Forest achieved the highest accuracy (91%)**, making it the most effective model in this case.
- **SVM performed well (79%)** and provides a robust alternative.
- **Linear Regression had the lowest accuracy (73%)**, likely due to the non-linear nature of the data.

## Conclusion

This study highlights the importance of choosing the right regression model for car price prediction. Random Forest outperformed the other models, demonstrating its robustness in handling complex datasets. The Gradio and Streamlit-based web apps provide user-friendly interfaces for real-time predictions, making the model accessible to non-technical users.

## Future Enhancements

- Implementing ensemble models for improved accuracy.
- Expanding the dataset to enhance model generalization.
- Integrating the model into a full-scale web or mobile application.

## Acknowledgments

- **Gradio** and **Streamlit** for enabling easy web deployment.
- **Scikit-learn** for robust machine learning tools.

## Contact

For any queries or collaboration opportunities, please contact:
- **Name:** Abdul Aziz Shaikh
- **Email:** shaikhaziz9920@gmail.com
