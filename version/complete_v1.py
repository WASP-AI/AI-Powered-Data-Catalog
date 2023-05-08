import os
import streamlit as st
import numpy as np
import pandas as pd
import re
from typing import List
import os
import pdfplumber
import mimetypes
from PyPDF2 import PdfReader
import docx2txt
from datetime import datetime, timedelta
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import UnstructuredPDFLoader, TextLoader, PyPDFLoader
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import CharacterTextSplitter
#from langchain.docstore.document import Document
#from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.prompts import PromptTemplate
#from langchain.chains.question_answering import load_qa_chain

st.set_page_config(page_title="Equinor Data Catalog", page_icon=":mag:")
# Display the image
st.image("equinor.png", width=200)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

def write_load(script):
    with open("Willdelete.txt", "w", encoding='utf-8') as f:
        f.write(script)
    
    loader = UnstructuredFileLoader("Willdelete.txt")
    docs= loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=750, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)

    embeddings = HuggingFaceHubEmbeddings()
    db = Chroma.from_documents(texts, embeddings)
    retri=db.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    return retri

def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page_number in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_number]
        text += page.extract_text()
    return text




def qa(q, retriever):
    
    flan_ul2 = HuggingFaceHub(
        repo_id="google/flan-ul2", model_kwargs={"temperature": 0.1, "max_length": 510}
    )

    prompt_template = """ 
    Given the following context please answer questions related to specific sections/subheaders: Timeliness, accuracy, uniqueness, latency, currency, description, additional information, and Information owner. 
    The questions will target a specific section/subheader 
    If the requested subheader is not found, return an appropriate message. 
    Please ensure that each output is clearly labeled and presented in a user-friendly manner.
    If the information can not be found in the context given, just say "i dont know".
    Don't try to make up an answer. Make sure to only use the context given in to answer the question.
    {context}
    Question: {question}
    Answer in English:"""

    PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": PROMPT}
    qaa = RetrievalQA.from_chain_type(llm=flan_ul2, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)
    
    query=q
    return qaa.run(query)


# Create the search bar
def clean_text(tet: str) -> str:
    # Replace multiple whitespaces with a single space
    tet = re.sub(r'\s+', ' ', tet)
    # Replace newline characters with a space
    tet = tet.replace('\n', ' ')
    return tet



def answer_questions(text, file_name):
    question_answers = {
        "File Name": file_name,
        "How often is the data updated?": "N/A",
        "Is additional information defined?": "N/A",
        "Is accuracy defined?": "N/A",
        "Is uniqueness defined?": "N/A",
        "Is timeliness defined?": "N/A",
        "Is information owner defined?": "N/A",
        "Is walid defined?": "N/A",
    }
    score = 0

    update_pattern = re.compile(r"(daily|weekly|monthly|annually|quarterly|every \d+ (days?|weeks?|months?|years?))", re.IGNORECASE)
    update_match = update_pattern.search(text)
    if update_match:
        question_answers["How often is the data updated?"] = update_match.group()

    keywords = ["additional information", "accuracy", "uniqueness", "timeliness", "information owner", "walid"]
    for keyword in keywords:
        keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        keyword_match = keyword_pattern.search(text)
        if keyword_match:
            question_answers[f"Is {keyword} defined?"] = "✔️"
            score += 1
        else:
            question_answers[f"Is {keyword} defined?"] = "❌"

    return question_answers, score

# Create the search bar
def clean_text(tet: str) -> str:
    # Replace multiple whitespaces with a single space
    tet = re.sub(r'\s+', ' ', tet)
    # Replace newline characters with a space
    tet = tet.replace('\n', ' ')
    return tet

# User inputs
files = st.file_uploader("Choose a file", accept_multiple_files=True, type=['txt','docx','pdf'])
question = st.text_input('Enter a question:')

# Define your list of questions
questions_list = ["What is Timeliness defined as in the document?", "Is latency defined within the timeliness section?",
                  "Is Uniqueness defined in the document?",
                  "What is Uniqueness defined as in the document?",
                  "Are there any duplicates in the data set?",
                  "What key value is used to ensure uniqueness for each entity in the data set?",
                  "Should users be aware of any quality issues related to Uniqueness that have occurred?", 
                  "Is the data from the IEA updated regularly each month?",
                  "Does the application check for updates every hour?",
                  "Is the time delay considered insignificant for the current use of the data?",]

if question:
    questions_list.append(question)

# Check if the session state already has questions_and_answers attribute; if not, create an empty dictionary
if "questions_and_answers" not in st.session_state:
    st.session_state["questions_and_answers"] = {}

# Process the uploaded files in a loop and check for the document type to process each document
if files:
    all_answers = []
    total_score = 0
    max_score_per_file = 6
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
            text = extract_text_from_pdf(file)
            metadata = PdfReader(file).metadata
            last_modified_str = metadata.get('/ModDate', 'D:19000101000000')
            last_modified_date = datetime.strptime(last_modified_str[2:16], '%Y%m%d%H%M%S')
            time_difference = datetime.now() - last_modified_date
            is_recent = time_difference <= timedelta(days=30)
        if is_recent:
            b="The PDF document was last modified within the past 30 days. Static"
        else:
            b="The PDF document was last modified more than 30 days ago. Volatile"
        textt=clean_text(text)

        answers, score = answer_questions(textt, file.name)
        answers["Is the data static or volatile?"] = b
        answers["Score"] = f"{score}/{max_score_per_file}"
        total_score += score

        # Add file size and file type
        file_size = os.path.getsize(file.name) / (1024 * 1024)
        file_type, _ = mimetypes.guess_type(file.name)
        answers["File Size"] = f"{file_size:.2f} MB"
        answers["File Type"] = file_type

        all_answers.append(answers)

        if file.name not in st.session_state["questions_and_answers"]:
            st.session_state["questions_and_answers"][file.name] = {}

        dbb = write_load(text)

        # Iterate through all predefined questions
            # Iterate through all predefined questions
        for predefined_question in questions_list:
            # Get the answer for the current file
            answer = qa(predefined_question, dbb)

            # Store the answer in the session state
            st.session_state["questions_and_answers"][file.name][predefined_question] = answer

    # Create a DataFrame from the all_answers list and transpose it for display
    st.write("Answers:")
    answer_df = pd.DataFrame(all_answers)
    transposed_df = answer_df.T

    st.write(
        transposed_df.style.set_table_styles(
            [
                {"selector": "table", "props": [("width", "100%"), ("table-layout", "auto")]},
                {"selector": "th, td", "props": [("word-wrap", "break-word"), ("white-space", "pre-wrap")]},
            ]
        )
    )
    max_total_score = len(files) * max_score_per_file
    st.write(f"Total Score: {total_score}/{max_total_score}")
    # Display the image
  

    # Display the search bar and predefined questions
    st.write('Data Description:')
    st.write("Please select one or more files and enter your questions to search for specific information.")
    st.write("The predefined questions are: What is Timeliness defined as in the document?, Is latency defined within the timeliness section?, Is Uniqueness defined in the document?, What is Uniqueness defined as in the document?, Are there any duplicates in the data set?, What key value is used to ensure uniqueness for each entity in the data set?, Should users be aware of any quality issues related to Uniqueness that have occurred?, Is the data from the IEA updated regularly each month?, Does the application check for updates every hour?, Is the time delay considered insignificant for the current use of the data?")

    # Display the results if any
    if "questions_and_answers" in st.session_state:
        df_results = pd.DataFrame(st.session_state["questions_and_answers"]).T
        if not df_results.empty:
            st.write("Answers:")
            st.table(df_results)
