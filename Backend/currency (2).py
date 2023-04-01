import os
import streamlit as st
import transformers
import torch
import streamlit as st
import numpy as np
import pandas as pd
from io import StringIO
import re
import PyPDF4
from typing import List
import os
import docx2txt
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredDocxLoader


def qa(q, c):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
    from langchain.llms import HuggingFaceHub
    from langchain.embeddings import HuggingFaceHubEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain
    from langchain.chains import VectorDBQA
    from langchain.document_loaders import UnstructuredFileLoader
    from langchain.prompts import PromptTemplate

    with open('Willdelete.txt', 'w') as f:
        f.write(c)

    loader = UnstructuredFileLoader("Willdelete.txt")
    docs= loader.load()


    flan_ul2 = HuggingFaceHub(
        repo_id="google/flan-ul2", model_kwargs={"temprature": 0.1, "max_new_tokens": 256}
    )


    text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=0)
    texts = text_splitter.split_text(docs[0].page_content)

    embeddings = HuggingFaceHubEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])

    prompt_template = """
    Given the following context containing multiple subheaders and their corresponding information, make sure to extract and print out the information under a specific subheader when requested. 
    The user will provide the name of the subheader they are interested in, and the you should output the corresponding information. 
    You should also handle variations in the subheader names and document formatting. 
    If the requested subheader is not found, return an appropriate message. 
    Please ensure that each output is clearly labeled and presented in a user-friendly manner.

    It is very important to make sure that you include everything under each subheader. 
    -----------------------------------------------------------------------------------
    For example: 
    (if the input is:
    
    What is currency? 
    Output will be:

    • Currency: is a good thing 
    ---------------------------------------------------------------------------------------
    Have this in mind when looking for answers to questions.
    ------------------------------------------------------------------------------------------------------------------------
    Use the following pieces of context to answer the question at the end. If the information can not be found in the context given, just say "dont know".
    Don't try to make up an answer. Make sure to only use the context given in {summaries} to answer the question. Look only for where it says Currency.
    if the document has a section where it is written currency print out  the rest of the header.


    {summaries}

    Question: {question}
    Answer in English:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question"]
    )

    chain = load_qa_with_sources_chain(flan_ul2, chain_type="stuff", prompt=PROMPT)
    
    docs = docsearch.similarity_search(question)
    return chain({"input_documents": docs, "question": q}, return_only_outputs=True)


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

st.set_page_config(page_title="Equinor Data Catalog", page_icon=":mag:")

# Get a list of uploaded files
files = get_uploaded_files()

# Create the search bar


# Show live suggestions based on the search term as the user types
suggestions = [file for file in files if file.endswith(('.txt', '.pdf'))]
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
results = []

# Process the uploaded files in a loop and checks for document type to process each document for the document type
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
            pdf_reader = PyPDF4.PdfFileReader(file)
            text = ""
            for page_num in range(pdf_reader.getNumPages()):
                page = pdf_reader.getPage(page_num)
                text += page.extractText()

        # Get the answer for the current file
        answer = qa(question, text)
        # Append the current results to the results list
        results.append({'File Name': file.name, question : answer})

# Convert the results to a pandas dataframe
df_results = pd.DataFrame(results)

# Display the results in a table
if not df_results.empty:
    st.write('Data Description:')
    st.table(df_results)


# Display the image
st.image("equinor.png", width=200)