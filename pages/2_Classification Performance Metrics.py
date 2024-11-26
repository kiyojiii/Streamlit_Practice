import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    log_loss, roc_auc_score, roc_curve
)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


# Function to load the dataset
@st.cache_data
def load_data(uploaded_file):
    dataframe = pd.read_csv(uploaded_file)
    return dataframe


# Function to preprocess data
def preprocess_data(dataframe):
    label_encoders = {}
    for column in dataframe.columns:
        if dataframe[column].dtype == 'object':
            encoder = LabelEncoder()
            dataframe[column] = encoder.fit_transform(dataframe[column])
            label_encoders[column] = encoder
    return dataframe, label_encoders


# Function to calculate log loss
def calculate_log_loss(model, X, Y, cv_method):
    log_loss_values = []
    for train_index, test_index in cv_method.split(X, Y):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        model.fit(X_train, Y_train)
        Y_prob = model.predict_proba(X_test)[:, 1]

        if len(np.unique(Y_test)) > 1:
            fold_log_loss = log_loss(Y_test, Y_prob)
            log_loss_values.append(fold_log_loss)
        else:
            log_loss_values.append(np.nan)

    log_loss_values = [loss for loss in log_loss_values if not np.isnan(loss)]
    return log_loss_values


# Function to evaluate the model
def evaluate_model(dataframe, method):
    dataframe, label_encoders = preprocess_data(dataframe)
    X = dataframe.iloc[:, :-1].values  # Features
    Y = dataframe.iloc[:, -1].values  # Target

    if method == "K-Fold Cross Validation":
        num_folds = st.sidebar.slider("Number of Folds (K-Fold):", 2, 20, 10)
        cv_method = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    elif method == "Leave One Out Cross Validation":
        cv_method = LeaveOneOut()

    model = LogisticRegression(max_iter=500, class_weight='balanced')

    st.sidebar.subheader("Display Options")
    # Sidebar checkboxes for metrics (automatically enabled)
    display_accuracy = st.sidebar.checkbox("Show Classification Accuracy", value=True)
    display_log_loss = st.sidebar.checkbox("Show Log Loss", value=True)
    display_conf_matrix = st.sidebar.checkbox("Show Confusion Matrix", value=True)
    display_class_report = st.sidebar.checkbox("Show Classification Report", value=True)
    display_roc = st.sidebar.checkbox("Show ROC Curve", value=True)

    with st.spinner(f"Running {method}..."):
        results = cross_val_score(model, X, Y, cv=cv_method, scoring="accuracy")
        log_loss_values = calculate_log_loss(model, X, Y, cv_method)

        y_pred = model.fit(X, Y).predict(X)
        matrix = confusion_matrix(Y, y_pred)
        report = classification_report(Y, y_pred, output_dict=True)

    # Classification Accuracy
    if display_accuracy:
        st.subheader("Classification Accuracy")
        st.write(f"Accuracy: {results.mean() * 100:.2f}% ± {results.std() * 100:.2f}%")

        # Plot Classification Accuracy
        if method == "K-Fold Cross Validation":
            plt.figure(figsize=(10, 5))
            plt.boxplot(results)
            plt.title(f'{method} Accuracy')
            plt.ylabel('Accuracy')
            plt.xticks([1], [f'{num_folds}-Fold'])
            st.pyplot(plt)

    # Log Loss
    if display_log_loss and log_loss_values:
        st.subheader("Log Loss")
        st.write(f"Mean Log Loss: {np.mean(log_loss_values):.3f} ± {np.std(log_loss_values):.3f}")

        # Plotting Log Loss for Each Fold
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(log_loss_values) + 1), log_loss_values, marker='o', linestyle='-')
        plt.title('Log Loss for Each Fold of Cross-Validation')
        plt.xlabel('Fold Number')
        plt.ylabel('Log Loss')
        plt.xticks(range(1, len(log_loss_values) + 1))
        plt.grid()
        st.pyplot(plt)

    # Confusion Matrix
    if display_conf_matrix:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        st.pyplot(fig)

    # Classification Report
    if display_class_report:
        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).transpose()
        st.markdown("""
        <style>
            .report-table {
                font-size: 18px;
                width: 100%;
                text-align: center;
            }
            .report-table th {
                font-size: 20px;
                padding: 10px;
                background-color: #f0f0f0;
            }
            .report-table td {
                font-size: 18px;
                padding: 10px;
            }
        </style>
        """, unsafe_allow_html=True)
        st.write(report_df.to_html(classes="report-table", index=True), unsafe_allow_html=True)

    # ROC Curve
    if display_roc:
        st.subheader("ROC Curve")
        auc_scores = []
        for train_index, test_index in cv_method.split(X, Y):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            model.fit(X_train, Y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            if len(np.unique(Y_test)) > 1:
                auc_score = roc_auc_score(Y_test, y_prob)
                auc_scores.append(auc_score)

        if auc_scores:
            fpr, tpr, _ = roc_curve(Y_test, y_prob)
            plt.figure(figsize=(10, 6))
            plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {np.mean(auc_scores):.2f})")
            plt.plot([0, 1], [0, 1], color="red", linestyle="--")
            plt.title("ROC Curve")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            st.pyplot(plt)


# Main app
def main():
    st.sidebar.subheader("Classification Sampling Techniques")
    method = st.sidebar.radio("Choose Sampling Technique:", ["K-Fold Cross Validation", "Leave One Out Cross Validation"])
    st.title("Classification Performance Metrics")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file:
        dataframe = load_data(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(dataframe.head())
        evaluate_model(dataframe, method)
    else:
        st.write("Please upload a CSV file to proceed.")


if __name__ == "__main__":
    main()
