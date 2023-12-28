import streamlit as st

# Split the page into two columns
left_column, right_column = st.columns(2)

# Add content to the left column
with left_column:
    st.header("Left Side")
    # Add your content for the left side here

# Add content to the right column
with right_column:
    st.header("Method Functionality")  # Title for the method information

    def display_method_info():
        st.sidebar.title("Method Functionality")

 method_info = {
        "process_docx_text": {
            "Functionality": "Extracts text from a DOCX file.",
            "Parameters": "docx_file (DOCX file path)",
            "Output": "Extracted text from the DOCX file"
        },
        "extract_text_from_uploaded_image": {
            "Functionality": "Uses Pytesseract to extract text from an uploaded image.",
            "Parameters": "uploaded_image (image file), language (optional language for text extraction, default is 'eng')",
            "Output": "Extracted text from the uploaded image"
        },
        "process_docx_text_without_lists": {
            "Functionality": "Removes lists from DOCX text.",
            "Parameters": "docx_file (DOCX file path)",
            "Output": "Text without lists from the DOCX file"
        },
        "process_pdf_text_without_lists": {
            "Functionality": "Extracts text from a PDF file without lists.",
            "Parameters": "pdf_file (PDF file path)",
            "Output": "Extracted text from the PDF file without lists"
        },
        "process_txt_file": {
            "Functionality": "Reads and extracts text from a TXT file.",
            "Parameters": "txt_file (TXT file object)",
            "Output": "Text extracted from the TXT file"
        },
        "translate_text_with_google": {
            "Functionality": "Translates text using Google Translate.",
            "Parameters": "text (text to translate), target_language (language code for translation)",
            "Output": "Translated text"
        },
        "convert_text_to_speech": {
            "Functionality": "Converts text to speech (MP3 format).",
            "Parameters": "text (text to convert), output_file (output file path), language (language code for speech synthesis)",
            "Output": "MP3 audio file with the generated speech"
        },
        "get_binary_file_downloader_html": {
            "Functionality": "Generates a download link for a file.",
            "Parameters": "link_text (text for the download link), file_path (file path), file_format (file format)",
            "Output": "HTML download link for the file"
        },
        "convert_text_to_word_doc": {
            "Functionality": "Converts translated text to a Word document.",
            "Parameters": "text (translated text to convert), output_file (output file path)",
            "Output": "Word document containing the translated text"
        }

        for method_name, details in method_info.items():
            st.sidebar.subheader(method_name)
            for key, value in details.items():
                st.sidebar.text(f"{key}: {value}")
            st.sidebar.text("\n")

    def main():
        st.image("jangirii.png", width=300)
        st.title("Text Translation and Conversion to Speech ( MultiLingual )")

        if __name__ == "__main__":
            display_method_info()
            main()
