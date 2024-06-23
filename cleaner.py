import streamlit as st
import pandas as pd
from transformers import pipeline

# Initialize the TAPAS model
pipe = pipeline("table-question-answering", model="google/tapas-large-finetuned-wtq")

def main():
    st.title("Table Question Answering App")
    
    # File upload
    uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])
    
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("Here is your data:")
        st.write(df)
        
        question = st.text_input("Ask a question about the table:")
        
        if question:
            result = pipe(table=df, query=question)
            st.write("Answer:")
            st.write(result['answer'])
    
if __name__ == "__main__":
    main()
