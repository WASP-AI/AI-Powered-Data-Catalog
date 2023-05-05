import os
import streamlit as st
import numpy as np
import pandas as pd
import re
from typing import List
import os
import pdfplumber
import docx2txt
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredPDFLoader, TextLoader, PyPDFLoader
from datetime import datetime, timedelta
from PyPDF2 import PdfFileReader



st.set_page_config(page_title="Equinor Data Catalog", page_icon=":mag:")
# Display the image
st.image("equinor.png", width=200)

def qa(q, c):

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "INSERT_API_TOKEN"
    from langchain.llms import HuggingFaceHub
    from langchain.chains import RetrievalQA
    from langchain.embeddings import HuggingFaceHubEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import CharacterTextSplitter
    #from langchain.docstore.document import Document
    #from langchain.indexes.vectorstore import VectorstoreIndexCreator
    from langchain.prompts import PromptTemplate
    #from langchain.chains.question_answering import load_qa_chain

    with open("Willdelete.txt", "w", encoding='utf-8') as f:
        f.write(c)
    
    loader = UnstructuredFileLoader("Willdelete.txt")
    docs= loader.load()
    
   # loader= UnstructuredFileLoader()
    
    flan_ul2 = HuggingFaceHub(
        repo_id="google/flan-ul2", model_kwargs={"temperature": 0.1, "max_length": 510}
    )

    text_splitter = CharacterTextSplitter(chunk_size=750, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceHubEmbeddings()

    db = Chroma.from_documents(texts, embeddings)
    retri=db.as_retriever(search_type="similarity", search_kwargs={"k": 1})

    prompt_template = """
    Given the following context {context} containing multiple subheaders and their corresponding information, make sure to extract and print out the information under a specific subheader when requested. 
    The user will provide the name of the subheader they are interested in, and the you should output the corresponding information. 
    You should also handle variations in the subheader names and document formatting. 
    If the requested subheader is not found, return an appropriate message. 
    Please ensure that each output is clearly labeled and presented in a user-friendly manner.
    If the information can not be found in the context given, just say "i dont know".
    Don't try to make up an answer. Make sure to only use the context given in {context} to answer the question.

    {context}

    Question: {question}
    Answer in English:"""

    PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    qaa = RetrievalQA.from_chain_type(llm=flan_ul2, chain_type="stuff", retriever=retri)
    
    query=q
    #docss = docsearch.get_relevant_documents(q)
    return qaa.run(query)


# Define a function to get a list of uploaded files in the current directory
def get_uploaded_files():
    files = []
    for item in os.listdir("."):
        if os.path.isfile(item):
            files.append(item)
    return files

# Define a function to read the contents of a file
def read_file(file):
    with open(file, "r") as f:
        content = f.read()
    return content


# Get a list of uploaded files
filess = get_uploaded_files()

# Create the search bar
def clean_text(text: str) -> str:
    # Replace multiple whitespaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Replace newline characters with a space
    text = text.replace('\n', ' ')
    return text


# Show live suggestions based on the search term as the user types
suggestions = [file for file in filess if file.endswith(('.txt', '.pdf'))]
# Create a multiselect widget to select files based on suggestions
selected_files = st.multiselect("Search:", suggestions)

# Display the selected files and their contents
if selected_files:
    st.write("Selected files:")
    for file in selected_files:
        st.write(file)
        content = read_file(file)
        st.text(content)

# User inputs

files = st.file_uploader("Choose a file", accept_multiple_files=True, type=['txt','docx','pdf'])
question = st.text_input('Enter a question:')

# Define your list of questions
questions_list = ["What is under the Timeliness section in the context?", "Is latency defined within the timeliness section?", 
                  "Is Uniqueness defined in the document?", 
                  "What is Uniqueness defined as in the document?", 
                  "Are there any duplicates in the data set?", 
                  "What key value is used to ensure uniqueness for each entity in the data set?", 
                  "Should users be aware of any quality issues related to Uniqueness that have occurred?", "Is the data from the IEA updated regularly each month?",
                  "Does the application check for updates every hour?",
                  "Is the time delay considered insignificant for the current use of the data?"]

if question:
    questions_list.append(question)
# Check if the session state already has questions_and_answers attribute; if not, create an empty dictionary
if "questions_and_answers" not in st.session_state:
    st.session_state["questions_and_answers"] = {}

# Process the uploaded files in a loop and check for the document type to process each document
if files:
    for file in files:
        file_extension = file.name.split('.')[-1]
        text = ''
        if file_extension == 'txt':
            content = file.read().decode('utf-8')
            text += content
        elif file_extension == 'docx':
            content = docx2txt.process(file)
            text += content
        elif file_extension == 'pdf':
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
            pdf_reader = PdfFileReader(file)
            metadata = pdf_reader.getDocumentInfo()
            last_modified_str = metadata.get('/ModDate', 'D:19000101000000')
            last_modified_date = datetime.strptime(last_modified_str[2:16], '%Y%m%d%H%M%S')
            time_difference = datetime.now() - last_modified_date
            is_recent = time_difference <= timedelta(days=30)

            if is_recent:
                b="The PDF document was last modified within the past 30 days. Static"
            else:
                b="The PDF document was last modified more than 30 days ago. Volatile"
        textt=clean_text(text)

        if file.name not in st.session_state["questions_and_answers"]:
            st.session_state["questions_and_answers"][file.name] = {}

        # Iterate through all predefined questions
        ##textt=clean_text(text)
        ###st.write(textt)
        st.session_state["questions_and_answers"][file.name]["Is the data static or volatile?"] = b
        for predefined_question in questions_list:
            # Get the answer for the current file
            answer = qa(predefined_question, text)

            # Store the answer in the session state
            st.session_state["questions_and_answers"][file.name][predefined_question] = answer

# Convert the results to a pandas dataframe
df_results = pd.DataFrame(st.session_state["questions_and_answers"]).T

            # Display the results in a table
if not df_results.empty:
                st.write('Data Description:')
                st.table(df_results)