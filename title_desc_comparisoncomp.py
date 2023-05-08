import os
import streamlit as st
import transformers
import torch
import numpy as np
import pandas as pd
from io import StringIO
import re
import PyPDF4
from typing import List
import os
import docx2txt
from transformers import BertTokenizer, BertForSequenceClassification, AutoTokenizer, AutoModel
from langchain.document_loaders import UnstructuredFileLoader
from langchain.document_loaders import PyPDFLoader


st.set_page_config(page_title="Equinor Data Catalog", page_icon=":mag:")

def qa(q, c):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = "insert api token"

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


    text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=0)
    texts = text_splitter.split_text(docs[0].page_content)

    embeddings = HuggingFaceHubEmbeddings()
    docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])

    prompt_template = """
    Given the following context containing
    
    The title is in the 2nd line of the PDF, it is before the I. 
    for example  oil external standard I Data Product I Collibra
    Please ensure that each output is clearly labeled and presented in a user-friendly manner.
    It is very important to make sure that you only include the Title. 
    -----------------------------------------------------------------------------------
    For example: 
    if the input is:
    
     • Is there a title for this 
  
      Output will be:
    • The title is: Jodi oil 
    ---------------------------------------------------------------------------------------
    Have this in mind when looking for answers to questions.
    ------------------------------------------------------------------------------------------------------------------------
    Use the following pieces of context to answer the question at the end. If the information can not be found in the context given, just say "dont know".
    Don't try to make up an answer. Make sure to only use the context given in {summaries} to answer the question.
    {summaries}
    Question: {question}
    Answer in English:"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["summaries", "question"]
    )

    chain = load_qa_with_sources_chain(flan_ul2, chain_type="stuff", prompt=PROMPT)
 
    docs = docsearch.similarity_search(e)
    a= chain({"input_documents": docs, "question": q}, return_only_outputs=True)
    b = dict(a)
    d=(b['output_text'])
    e=str(d)
    st.write(e)
    docs = docsearch.similarity_search(e)
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


# Get a list of uploaded files
files = get_uploaded_files()

# Create the search bar


# Show live suggestions based on the search term as the user types
#suggestions = [file for file in files if file.endswith(('.txt', '.pdf'))]
# Create a multiselect widget to select files based on suggestions
#selected_files = st.multiselect("Search:", suggestions)

# Display the selected files and their contents
#if selected_files:
 #   st.write("Selected files:")
  #  for file in selected_files:
   #     st.write(file)
    #    content = read_file(file)
     #   st.text(content)

# User inputs

files  = st.file_uploader("Choose a file", accept_multiple_files=True, type=['txt','docx','pdf'])
e = ('What is the title for this document?')
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
        answer = qa(e, text)
        # Append the current results to the results list
        results.append({'File Name': file.name, e : answer})

# Convert the results to a pandas dataframe
df_results = pd.DataFrame(results)

# Display the results in a table
if not df_results.empty:
    st.write('Data Description:')
    st.table(df_results)



## The lines after is for the description part

@st.cache_resource
def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    return tokenizer, model


def get_paragraphs(pdf_file, header):
    paragraphs = []
    pdf_reader = PyPDF4.PdfFileReader(pdf_file)
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        text = page.extractText()
        pattern = r"\n\s*{}".format(header)
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        match_positions = [match.start() for match in matches]
        match_positions.append(len(text))
        for i in range(len(match_positions) - 1):
            start = match_positions[i] + len(header) + 1
            end = match_positions[i+1]
            paragraphs.append(text[start:end])
    return paragraphs


def classify_paragraphs(paragraphs, tokenizer, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    predicted_labels = []
    for p in paragraphs:
        inputs = tokenizer.encode_plus(p, return_tensors='pt', padding=True, truncation=True)
        inputs.to(device)
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()[0]
        predicted_label = int(logits.argmax())
        predicted_labels.append(predicted_label)
    return predicted_labels




def comparison(file):
    global para
#    st.title("PDF Paragraph Classifier")
    uploaded_file=file
    if uploaded_file is not None:
        headers = ["Description", "Additional Information"]
        #Headers choice with a selectbox creates a tab that shows on the streamlit page use only the tuple headers_choice

        #header_choice = st.selectbox(headers, label_visibility="visible", disabled =True) 
        header_choice = (headers)
        paragraphs = get_paragraphs(uploaded_file, header_choice)
        tokenizer, model = load_bert_model()
        predicted_labels = classify_paragraphs(paragraphs, tokenizer, model)
        all_paragraphs = ''
        for i, p in enumerate(paragraphs):
            if predicted_labels[i] == 1:
                all_paragraphs += "**{}**\n\n".format(p)
            else:
                all_paragraphs += "{}\n\n".format(p)
        #st.write(all_paragraphs)
        para = all_paragraphs
#        desc=para
    #st.write(para)



    ##title description
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    model = AutoModel.from_pretrained("bert-base-cased")

    title = e
    title_desc = para



  



    inputs = tokenizer.encode_plus(title, return_tensors='pt', max_length=512, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    output = model(input_ids, attention_mask=attention_mask)
    last_hidden_state = output[0]
    pooled_output_title = torch.mean(last_hidden_state, 1)

    inputs = tokenizer.encode_plus(title_desc, return_tensors='pt', max_length=512, truncation=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    output = model(input_ids, attention_mask=attention_mask)
    last_hidden_state = output[0]
    pooled_output_desc = torch.mean(last_hidden_state, 1)

    pooled_output = torch.cat([pooled_output_title, pooled_output_desc], dim=0)
    similarity_score = torch.nn.functional.cosine_similarity(pooled_output[0], pooled_output[1], dim=0).item()


    avrundet_similarity_score= float("%.3f"%similarity_score)*100

    if not title:
        st.write("The title is left blank and should not be left blank, unless testing for limitations")
    if not title_desc:
        st.write("The description is left blank and should not be left blank, unless testing for limitations")


    if( title and title_desc) :
        return avrundet_similarity_score

if __name__ == '__main__':
    comparison(file)
    #Display the image and sliders in the sidebar
