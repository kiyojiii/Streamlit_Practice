import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    column_names = [
        'Date', 'Time', 'CO_GT', 'PT08_S1_CO', 'NMHC_GT', 'C6H6_GT', 'PT08_S2_NMHC',
        'Nox_GT', 'PT08_S3_Nox', 'NO2_GT', 'PT08_S4_NO2', 'PT08_S5_O3', 'T', 'RH', 'AH', 'CO_level'
    ]
    dataframe = pd.read_csv(uploaded_file, names=column_names, skiprows=1)
    return dataframe

# Function to handle non-numeric columns (e.g., encoding categorical data)
def handle_non_numeric(dataframe):
    label_encoders = {}
    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':  # If the column contains strings
            le = LabelEncoder()
            dataframe[column] = le.fit_transform(dataframe[column])
            label_encoders[column] = le
    return dataframe, label_encoders

# Function to compute metrics
def compute_metrics(model, X_train, X_test, Y_train, Y_test):
    # Train the model
    model.fit(X_train, Y_train)
    # Predict on test data
    Y_pred = model.predict(X_test)
    # Calculate metrics
    mse = mean_squared_error(Y_test, Y_pred)
    mae = mean_absolute_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)
    return mse, mae, r2

# Main app
def main():
    st.title("Linear Regression Performance Metrics")

    # Sidebar for toggles
    st.sidebar.subheader("Linear Regression Sampling Techniques")
    method = st.sidebar.radio(
        "Select the Sampling Technique:",
        ["Split Into Train and Test Sets", "Repeated Random Train-Test Splits"]
    )
    st.sidebar.subheader("Display Options")
    show_dataset_preview = st.sidebar.checkbox("Show Dataset Preview", value=True)
    show_mse = st.sidebar.checkbox("Display MSE", value=True)
    show_mae = st.sidebar.checkbox("Display MAE", value=True)
    show_r2 = st.sidebar.checkbox("Display R² Score", value=True)

    # Step 2: Upload CSV File
    uploaded_file = st.file_uploader("Upload your CSV file with air quality data", type=["csv"])

    if uploaded_file is not None:
        # Load the dataset
        st.write("Loading the dataset...")
        dataframe = load_data(uploaded_file)

        # Display first few rows of the dataset
        if show_dataset_preview:
            st.subheader("Dataset Preview")
            st.write(dataframe.head())

        # Handle missing values and invalid data
        dataframe = dataframe.replace(-200, np.nan).dropna()

        # Encode non-numeric columns
        dataframe, label_encoders = handle_non_numeric(dataframe)

        # Prepare the features and target variables
        X = dataframe.drop(columns=["T", "RH", "AH", "Date", "Time", "CO_level"])  # Features
        Y = dataframe[["T", "RH", "AH", "CO_level"]]  # Target variables

        if method == "Split Into Train and Test Sets":
            # Train-test split parameters
            test_size = 0.2
            seed = 42
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

            # Initialize the model
            model = LinearRegression()

            # Compute metrics
            mse, mae, r2 = compute_metrics(model, X_train, X_test, Y_train, Y_test)

            # Display results
            st.subheader("Model Evaluation: Train-Test Split")
            if show_mse:
                st.write(f"Mean Squared Error (MSE): {mse:.3f}")
            if show_mae:
                st.write(f"Mean Absolute Error (MAE): {mae:.3f}")
            if show_r2:
                st.write(f"R² Score: {r2:.3f}")

        elif method == "Repeated Random Train-Test Splits":
            # ShuffleSplit parameters
            n_splits = 10  # Number of repetitions
            test_size = 0.2
            shuffle_split = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)

            # Initialize the model
            model = LinearRegression()

            mse_scores = []
            mae_scores = []
            r2_scores = []

            # Perform repeated random testing
            for train_index, test_index in shuffle_split.split(X, Y):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

                # Compute metrics
                mse, mae, r2 = compute_metrics(model, X_train, X_test, Y_train, Y_test)
                mse_scores.append(mse)
                mae_scores.append(mae)
                r2_scores.append(r2)

            # Display results
            st.subheader("Model Evaluation: Repeated Random Testing (ShuffleSplit)")
            if show_mse:
                st.write(f"Mean Squared Error (MSE): {np.mean(mse_scores):.3f} (+/- {np.std(mse_scores):.3f})")
            if show_mae:
                st.write(f"Mean Absolute Error (MAE): {np.mean(mae_scores):.3f} (+/- {np.std(mae_scores):.3f})")
            if show_r2:
                st.write(f"R² Score: {np.mean(r2_scores):.3f} (+/- {np.std(r2_scores):.3f})")

    else:
        st.write("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
