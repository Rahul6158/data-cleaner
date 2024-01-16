import streamlit as st
import pandas as pd
import numpy as np

# Function to load sample data
@st.cache
def load_sample_data():
    return pd.DataFrame({
        'Name': ['John', 'Jane', 'Bob'],
        'Age': [25, 30, 22],
        'Salary': [50000, 60000, 45000]
    })

# Main function
def main():
    st.title("Dataset Operations App")

    # Sidebar for file upload
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    # Load sample data if no file is uploaded
    if uploaded_file is None:
        st.warning("Please upload a CSV file.")
        data = load_sample_data()
    else:
        # Load the uploaded dataset
        try:
            data = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error: {e}")
            return

    # Display uploaded dataset
    st.subheader("Uploaded Dataset")
    st.write(data)

    # Basic operations
    st.subheader("Basic Operations")

    # Show summary statistics
    if st.checkbox("Show Summary Statistics"):
        st.write(data.describe())

    # Show correlation heatmap
    if st.checkbox("Show Correlation Heatmap"):
        st.write(data.corr())

    # Custom operations
    st.subheader("Custom Operations")

    # Example: Calculate mean salary
    if st.button("Calculate Mean Salary"):
        mean_salary = data['Salary'].mean()
        st.write(f"The mean salary is: {mean_salary}")

    # Example: Plot histogram of ages
    if st.button("Plot Histogram of Ages"):
        st.bar_chart(data['Age'].value_counts())

    # Add more custom operations based on your needs

if __name__ == "__main__":
    main()
