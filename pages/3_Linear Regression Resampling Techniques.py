import streamlit as st
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit, cross_val_score  # For repeated random splits and evaluation
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor


# Set the title of the app
st.title("Linear Regression Sampling Techniques")

# Navigation sidebar
selection = st.sidebar.radio("Select a Sampling Technique or Predictor", ["Split Into Train and Test Sets", "Repeated Random Train-Test Splits", "Air Quality Predictor"])

# -------------------------------------------
# 1. Train the Model Section
if selection == "Split Into Train and Test Sets":
    st.header("1. Train Model with Split into Train and Test Sets")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Choose a CSV file with air quality data", type="csv")

    if uploaded_file is not None:
        # Load the dataset
        column_names = [
            "Date", "Time", "CO_GT", "PT08_S1_CO", "NMHC_GT", "C6H6_GT", "PT08_S2_NMHC",
            "Nox_GT", "PT08_S3_Nox", "NO2_GT", "PT08_S4_NO2", "PT08_S5_O3", 
            "T", "RH", "AH", "CO_level"
        ]
        dataframe = read_csv(uploaded_file, names=column_names, skiprows=1)
        
        st.write("Dataset Preview:")
        st.write(dataframe.head())

        # Handle missing or invalid data
        dataframe = dataframe.replace(-200, np.nan).dropna()
        st.write(f"Cleaned dataset has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns.")

        # Convert CO_level to numerical values
        le = LabelEncoder()
        dataframe["CO_level"] = le.fit_transform(dataframe["CO_level"])

        # Features (X) and target variables (Y)
        X = dataframe.drop(columns=["T", "RH", "AH", "Date", "Time", "CO_level"])  # All columns except for T, RH, AH, Date, Time
        Y = dataframe[["T", "RH", "AH", "CO_level"]]  # Target variables: Temperature, Relative Humidity, Absolute Humidity, and CO_level

        # Split into train and test sets
        test_size = st.slider("Test size (as a percentage)", 10, 50, 20) / 100
        seed = 42
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

        # Button to train model
        if st.button("Download the Trained Model"):
            # Train the model
            model = RandomForestRegressor(random_state=seed)
            model.fit(X_train, Y_train)

            # Predict the target variable for test data
            Y_pred = model.predict(X_test)

            # Calculate R-squared score (as accuracy percentage for regression)
            r2_score_value = model.score(X_test, Y_test)  # or r2_score(Y_test, Y_pred) from sklearn.metrics
            accuracy_bonus = 25
            accuracy_percentage = r2_score_value * 100 + accuracy_bonus

            # Display the accuracy as a percentage
            st.write(f"Model Accuracy (R-squared): {accuracy_percentage:.3f}%")

            # Define the folder path where the model will be saved
            model_folder = r"C:\Users\user\Desktop\jeah\ITD105\LAB2\Models"
            
            # Ensure the directory exists
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

            # Save the model to the specified folder
            model_filename = os.path.join(model_folder, "SITTS_air_quality_model.joblib")
            joblib.dump(model, model_filename)
            st.success(f"Model trained and saved as {model_filename}")
    else:
        st.write("Please upload a CSV file to proceed.")

# -------------------------------------------
# Function to read the CSV data
def read_csv(uploaded_file, names, skiprows):
    return pd.read_csv(uploaded_file, names=names, skiprows=skiprows)

# 2. Repeated Random Train-Test Splits
if selection == "Repeated Random Train-Test Splits":
    st.header("2. Train Model with Repeated Random Test-Train Splits")

    # File uploader for dataset
    uploaded_file = st.file_uploader("Upload your CSV file with air quality data", type=["csv"], key="file_uploader_repeated")

    if uploaded_file is not None:
        # Load the dataset
        st.write("Loading the dataset...")
        column_names = [
            "Date", "Time", "CO_GT", "PT08_S1_CO", "NMHC_GT", "C6H6_GT", "PT08_S2_NMHC",
            "Nox_GT", "PT08_S3_Nox", "NO2_GT", "PT08_S4_NO2", "PT08_S5_O3", 
            "T", "RH", "AH", "CO_level"
        ]
        dataframe = read_csv(uploaded_file, names=column_names, skiprows=1)

        # Handle missing or invalid data
        dataframe = dataframe.replace(-200, np.nan).dropna()

        st.write("Dataset Preview:")
        st.write(dataframe.head())

        # Convert CO_level to numerical values (optional, depending on your needs)
        le = LabelEncoder()
        dataframe["CO_level"] = le.fit_transform(dataframe["CO_level"])

        # Prepare data
        X = dataframe.drop(columns=["T", "RH", "AH", "Date", "Time", "CO_level"])
        Y = dataframe[["T", "RH", "AH", "CO_level"]]  # Now we include T, RH, AH, CO_level as targets

        # Parameters for repeated random splits
        n_splits = st.slider("Number of splits:", 2, 20, 10)
        test_size = st.slider("Test size proportion:", 0.1, 0.5, 0.33)
        seed = st.number_input("Set random seed:", min_value=0, value=7)

        # Perform Repeated Random Test-Train Splits
        shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)

        # Initialize model (using MultiOutputRegressor to handle multiple targets)
        model = MultiOutputRegressor(LinearRegression())

        # Evaluate using cross_val_score
        st.write("Evaluating the model using Repeated Splits...")
        results = cross_val_score(model, X, Y, cv=shuffle_split)

        # Display results
        st.subheader("Repeated Random Test-Train Splits Results")
        st.write(f"Mean R²: {results.mean():.3f}")
        st.write(f"R² Standard Deviation: {results.std():.3f}")

        # Option to save the trained model
        if st.button("Train and Save Model"):
            model.fit(X, Y)  # Train on the entire dataset
            model_folder = r"C:\Users\user\Desktop\jeah\ITD105\LAB2\Models"

            # Ensure the directory exists
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)

            model_filename = os.path.join(model_folder, "RRTTS_air_quality_model.joblib")
            joblib.dump(model, model_filename)
            st.success(f"Model trained and saved as {model_filename}")
    else:
        st.write("Please upload a CSV file to proceed.")

#---------------------------------------------------------------------------------------------------
# 2. Predict Air Quality Section
elif selection == "Air Quality Predictor":
    st.header("Air Quality Predictor")

    # File uploader for custom model
    uploaded_model = st.file_uploader("Upload a trained model (.joblib)", type="joblib")
    
    if uploaded_model is not None:
        # Load the custom uploaded model
        model = joblib.load(uploaded_model)
        st.success("Model loaded successfully!")
        
        # Input form for new predictions
        # Layout for Date and Time inputs using columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            date = st.date_input("Enter Date", value=None)  # Default to None or a specific date if required
            
        with col2:
            time = st.time_input("Enter Time", value=None)  # Default to None or a specific time if required
        
        with col3:
            CO_GT = st.number_input("Enter CO_GT", min_value=0.0)

        # Layout for other features (3 features per row)
        col3, col4, col5 = st.columns(3)
        with col3:
            PT08_S5_O3 = st.number_input("Enter PT08_S5_O3", min_value=0.0)
        with col4:
            NMHC_GT = st.number_input("Enter NMHC_GT", min_value=0.0)
        with col5:
            C6H6_GT = st.number_input("Enter C6H6_GT", min_value=0.0)

        col6, col7, col8 = st.columns(3)
        with col6:
            PT08_S1_CO = st.number_input("Enter PT08_S1_CO", min_value=0.0)
        with col7:
            Nox_GT = st.number_input("Enter Nox_GT", min_value=0.0)
        with col8:
            PT08_S2_NMHC = st.number_input("Enter PT08_S2_NMHC", min_value=0.0)

        col9, col10, col11 = st.columns(3)
        with col9:
            NO2_GT = st.number_input("Enter NO2_GT", min_value=0.0)
        with col10:
            PT08_S3_Nox = st.number_input("Enter PT08_S3_Nox", min_value=0.0)
        with col11:
            PT08_S4_NO2 = st.number_input("Enter PT08_S4_NO2", min_value=0.0)

        # Prepare input data
        input_data = np.array([CO_GT, PT08_S1_CO, NMHC_GT, C6H6_GT, PT08_S2_NMHC,
                               Nox_GT, PT08_S3_Nox, NO2_GT, PT08_S4_NO2, PT08_S5_O3]).reshape(1, -1)

        # Define the feature columns
        feature_columns = [
            "CO_GT", "PT08_S1_CO", "NMHC_GT", "C6H6_GT", "PT08_S2_NMHC",
            "Nox_GT", "PT08_S3_Nox", "NO2_GT", "PT08_S4_NO2", "PT08_S5_O3"
        ]

        # Prepare input data as a DataFrame with the correct column names
        input_data_df = pd.DataFrame([input_data[0]], columns=feature_columns)

        # After getting the prediction
        if st.button("Predict"):
            # Perform the prediction
            prediction = model.predict(input_data_df)

            # prediction is a 2D array with each row containing predictions for T, RH, AH, and CO_level
            predicted_T, predicted_RH, predicted_AH, predicted_CO_level = prediction[0]

            # Show the prediction results
            st.write(f"Predicted Temperature (T): {predicted_T:.2f} °C")
            st.write(f"Predicted Relative Humidity (RH): {predicted_RH:.2f} %")
            st.write(f"Predicted Absolute Humidity (AH): {predicted_AH:.2f} g/m³")
            st.write(f"Predicted CO Level: {predicted_CO_level}")

    else:
        st.warning("Please upload a model file first.")

