import streamlit as st
def get_method_code(method_name):
    if method_name == "process_docx_text":
        return '''
        def process_docx_text(docx_file, skip_lists=True):
            if skip_lists:
                text = process_docx_text_without_lists(docx_file)
            else:
                text = docx2txt.process(docx_file)
            return text
        '''
    elif method_name == "extract_text_from_uploaded_image":
        return '''
        def extract_text_from_uploaded_image(uploaded_image, language='eng'):
            try:
                image = Image.open(uploaded_image)
                image = image.convert('RGB')
                text = pytesseract.image_to_string(image, lang=language)
                return text
            except Exception as e:
                return str(e)
        '''
    elif method_name == "process_docx_text_without_lists":
        return '''
        def process_docx_text_without_lists(docx_file):
            doc = Document(docx_file)
            text = ""
            for paragraph in doc.paragraphs:
                if not paragraph.style.name.startswith('•'):
                    text += paragraph.text + '\\n'
            return text
        '''
    elif method_name == "process_pdf_text_without_lists":
        return '''
        def process_pdf_text_without_lists(pdf_file):
            pdf_text = ""
            try:
                with st.spinner("Extracting text from PDF..."):
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    num_pages = len(pdf_reader.pages)
                    for page_number in range(num_pages):
                        page = pdf_reader.pages[page_number]
                        pdf_text += page.extract_text()
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            return pdf_text
        '''
    elif method_name == "process_txt_file":
        return '''
        def process_txt_file(txt_file):
            txt_text = txt_file.read()
            text = txt_text.decode('utf-8')
            return text
        '''
    elif method_name == "translate_text_with_google":
        return '''
        def translate_text_with_google(text, target_language):
            google_translator = GoogleTranslator()
            max_chunk_length = 500
            translated_text = ""
            for i in range(0, len(text), max_chunk_length):
                chunk = text[i:i + max_chunk_length]
                translated_chunk = google_translator.translate(chunk, dest=target_language).text
                translated_text += translated_chunk
            return translated_text
        '''
    elif method_name == "convert_text_to_speech":
        return '''
        def convert_text_to_speech(text, output_file, language='en'):
            if text:
                supported_languages = list(language_mapping.keys())
                if language not in supported_languages:
                    st.warning(f"Unsupported language: {language}")
                    return
                tts = gTTS(text=text, lang=language)
                tts.save(output_file)
        '''
    elif method_name == "get_binary_file_downloader_html":
        return '''
        def get_binary_file_downloader_html(link_text, file_path, file_format):
            with open(file_path, 'rb') as f:
                file_data = f.read()
            b64_file = base64.b64encode(file_data).decode()
            download_link = f'<a href="data:{file_format};base64,{b64_file}" download="{os.path.basename(file_path)}">{link_text}</a>'
            return download_link
        '''
    elif method_name == "convert_text_to_word_doc":
        return '''
        def convert_text_to_word_doc(text, output_file):
            doc = Document()
            doc.add_paragraph(text)
            doc.save(output_file)
        '''



def display_method_info(method_info):
    st.header("Method Functionality")
    for method_name, details in method_info.items():
        st.subheader(method_name)
        for key, value in details.items():
            st.text(f"{key}: {value}")
        st.text("\n")

def display_method_code(method_names):
    st.header("code of the method")
    for method_name in method_names:
        method_code = get_method_code(method_name)
        st.code(method_code, language='python')


def display_method_info():
    st.header("Method Functionality")

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
    }

    for method_name, details in method_info.items():
        st.subheader(method_name)
        for key, value in details.items():
            st.text(f"{key}: {value}")
        st.text("\n")

def main():
    # Split the page into two columns
    left_column, right_column = st.beta_columns(2)


    # List of method names
    method_names = [
        "process_docx_text",
        "extract_text_from_uploaded_image",
        "process_docx_text_without_lists",
        "process_pdf_text_without_lists",
        "process_txt_file",
        "translate_text_with_google",
        "convert_text_to_speech",
        "get_binary_file_downloader_html",
        "convert_text_to_word_doc"
    ]

    # Add content to the left column
    with left_column:
        display_method_code(method_names)
        # Add your content for the left side here

    # Add content to the right column
    with right_column:
        display_method_info()

if __name__ == "__main__":
    main()
