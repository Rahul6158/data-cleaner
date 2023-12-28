import streamlit as st
import pandas as pd
import base64

def null_removal_page():
    st.title("Dataset Handler - Null Value Removal")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)

        # Display basic details about the dataset
        st.subheader("Dataset Information:")
        st.write(f"Number of Rows: {uploaded_data.shape[0]}")
        st.write(f"Number of Columns: {uploaded_data.shape[1]}")
        st.write("Preview of the Dataset:")
        st.write(uploaded_data.head())

        # Show null value counts
        st.subheader("Null Values in the Dataset:")
        st.write(uploaded_data.isnull().sum())

        # Button to clear null values
        if st.button("Clear Null Values"):
            modified_data = uploaded_data.dropna()  # Remove rows with null values

            # Display modified data
            st.title("Modified Dataset:")
            st.write(modified_data.head())

            # Download the modified dataset as CSV
            csv_file = modified_data.to_csv(index=False)
            b64 = base64.b64encode(csv_file.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="modified_dataset.csv">Download Modified Dataset</a>'
            st.markdown(href, unsafe_allow_html=True)

def main():
    null_removal_page()

if __name__ == "__main__":
    main()
