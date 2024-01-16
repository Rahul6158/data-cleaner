# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from scipy.stats import ttest_ind

# Load sample data
@st.cache
def load_data():
    data = sns.load_dataset("iris")
    return data

# Sidebar for feature selection
st.sidebar.header("Choose Features")
selected_features = st.sidebar.multiselect(
    "Select features for analysis",
    ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
)

# Load data
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

# Machine Learning
st.subheader("Machine Learning")
target_variable = "species"
features = [feature for feature in selected_features if feature != target_variable]

X_train, X_test, y_train, y_test = train_test_split(
    data[features], data[target_variable], test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

if st.checkbox("Show Model Performance"):
    st.write(f"Accuracy: {accuracy_score(y_test, predictions)}")
    st.write(f"Classification Report:\n{classification_report(y_test, predictions)}")

# Text Mining and NLP
st.subheader("Text Mining and NLP")
text_data = st.text_area("Enter text for analysis")
if text_data:
    # Word Cloud
    st.subheader("Word Cloud")
    wordcloud = WordCloud().generate(text_data)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot()

    # Text Vectorization
    vectorizer = CountVectorizer()
    text_vectorized = vectorizer.fit_transform([text_data])
    st.write("Text Vectorization:")
    st.write(pd.DataFrame(text_vectorized.toarray(), columns=vectorizer.get_feature_names_out()))

# Statistical Analysis
st.subheader("Statistical Analysis")
variable1 = st.selectbox("Select Variable 1 for T-Test", selected_features)
variable2 = st.selectbox("Select Variable 2 for T-Test", selected_features)

if st.button("Run T-Test"):
    t_statistic, p_value = ttest_ind(data[variable1], data[variable2])
    st.write(f"T-Statistic: {t_statistic}")
    st.write(f"P-Value: {p_value}")

# Custom Features
st.subheader("Custom Features")

# Your custom features can be added here based on your requirements.

# Ensure to run the Streamlit app using streamlit run your_app_name.py in the terminal.
