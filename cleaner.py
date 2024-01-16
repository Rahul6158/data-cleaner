# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load sample data
@st.cache
def load_data():
    data = sns.load_dataset("iris")
    return data

# Sidebar for feature selection
st.sidebar.header("Choose Features")
selected_features = st.sidebar.multiselect(
    "Select features for analysis",
    ["sepal_length", "sepal_width", "petal_length", "petal_width"]
)

# Load data
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    # Use sample data if no file is uploaded
    data = load_data()

# Display DataFrame
st.subheader("Display DataFrame")
st.write(data[selected_features])

# Exploratory Data Analysis
st.subheader("Exploratory Data Analysis")
if st.checkbox("Show Summary Statistics"):
    st.write(data[selected_features].describe())

if st.checkbox("Show Correlation Heatmap"):
    correlation_matrix = data[selected_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()

# Outlier Detection
st.subheader("Outlier Detection")
if st.checkbox("Show Boxplot for Outliers"):
    for feature in selected_features:
        st.write(f"Outliers in {feature}")
        sns.boxplot(x=data[feature])
        st.pyplot()

# Machine Learning - Linear Regression
st.subheader("Machine Learning - Linear Regression")

# Select the target variable
target_variable = st.selectbox("Select the target variable for Linear Regression", selected_features)

if st.button("Run Linear Regression"):
    X_train, X_test, y_train, y_test = train_test_split(
        data[selected_features], data[target_variable], test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    st.write("Linear Regression Model Performance:")
    st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")
    st.write(f"R-squared: {r2_score(y_test, predictions)}")

    # Display coefficients
    st.write("Model Coefficients:")
    st.write(pd.DataFrame({"Feature": selected_features, "Coefficient": model.coef_}))

# Custom Features
st.subheader("Custom Features")

# Your custom features can be added here based on your requirements.

# Ensure to run the Streamlit app using streamlit run your_app_name.py in the terminal.
