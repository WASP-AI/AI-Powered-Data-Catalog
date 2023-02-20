import streamlit as st
import pandas as pd
import transformers
import numpy as np
import torch
from io import StringIO
import torch
import Levenshtein


def tex_clar():
    # Load the pre-trained language model for text classification
    model = transformers.pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

        # Add a title and description
    st.title("Data Description Clarity Checker")
    st.write("Upload a data file and write a description for the data. Then, check if the description is easy to understand.")

        # Add a file uploader
    upl_file = st.file_uploader("Choose a file")

        # Add a text input box for user to enter description
    desc = st.text_input("Enter data description here")

        # Check if description is easy to understand


    if desc:
        result = model("distilbert-base-uncased-finetuned-sst-2-english")[0]
        label = result["label"]
        score = result["score"]

        st.write (f"Label: {label}, Score: {score}")

        # If a file is uploaded, display its content in a table
    if upl_file:
        data = pd.read(upl_file)
        st.write("Uploaded data:")
        st.write(data)

        # If a description is entered, display it
    if desc:
        st.write("Data description:")
        st.write(desc)

                # Display the result





def dat_sor():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import torch
    from io import StringIO
    import transformers
    import re
    import PyPDF2
    from typing import List
    import os
    import docx2txt
    from transformers import AutoTokenizer, BertForQuestionAnswering, AutoModelForQuestionAnswering

    tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
    model = BertForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

    choice = st.radio(
        "What type of input:",
        ('Live input', 'Document'))

    if choice == 'Live input':
        st.write('You selected Live input.')
        text1=st.text_area("Enter Text Here")
        question1=st.text_area("Enter question Here")
        question = question1
        text = text1

        inputs = tokenizer(question, text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        answer_start_score = outputs.start_logits.max(dim=1).values
        answer_end_score = outputs.end_logits.max(dim=1).values

        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        st.write(tokenizer.decode(predict_answer_tokens, skip_special_tokens=True))

    else:
            # Load BERT Question Answering model
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        model = AutoModelForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

        # Function to preprocess and tokenize the text
        def preprocess_text(text: str):
            text = re.sub(r'[\n]+', r' ', text)
            text = re.sub(r'[^\x00-\x7F]+', r' ', text)
            text = re.sub(r'[^\w\s]+', r' ', text)
            text = text.strip()
            input_ids = tokenizer.encode(text, return_tensors='pt')
            return input_ids

        # Function to get the answer to the question
        def get_answer(text, question):
            inputs = tokenizer(question, text, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                answer_start_scores, answer_end_scores = outputs[0], outputs[1]

            answer_start = torch.argmax(answer_start_scores)
            answer_end = torch.argmax(answer_end_scores) + 1

            answer = tokenizer.decode(input_ids[0][answer_start:answer_end])
            return answer

        def main():
            # User inputs
            st.title('Question Answering with BERT')
            question = st.text_input('Enter a question:')
            files = st.file_uploader("Choose a file", accept_multiple_files=True, type=['txt','docx','pdf'])

            # Process the uploaded files
            if files:
                st.write('Answer:')
                for file in files:
                    file_extension = file.name.split('.')[-1]
                    text = ''
                    if file_extension == 'txt':
                        file_content = file.read().decode('utf-8')
                        text += file_content
                    elif file_extension == 'docx':
                        file_content = docx2txt.process(file)
                        text += file_content
                    elif file_extension == 'pdf':
                        pdf = PyPDF2.PdfFileReader(file)
                        for page in range(pdf.numPages):
                            text += pdf.getPage(page).extractText()

                    # Get the answer for the current file
                    answer = get_answer(text, question)
                    # Display the answer
                    
                    st.write('\n')
                    if answer:
                        st.write(f'Answer for {file.name}: {answer}')
                        
                    else:
                        st.write(f'No answer found for {file.name}.')


        if __name__ == '__main__':
            main()
            st.write("")


def mat_tit():
    ##title description
    from transformers import AutoTokenizer, AutoModel



    def number_to_word(text): 
        ##creating a dictonary to convert numerical values to text (if a user inputs '1' then it will be changed to one, etc.)
        words = {
            "0": "zero",

            "1": "one",
            
            "2": "two",
            
            "3": "three",
            
            "4": "four",
            
            "5": "five",
            
            "6": "six",
            
            "7": "seven",
            
            "8": "eight",
            
            "9": "nine",
        }
        for key, value in words.items(): 
            ##  running in a for loop to check for any numerical values and changing them to one two trhree
            ## replacing the keys and value 
            text = text.replace(key, value)
        return text

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")

    title = st.text_input ("Enter title: ")
    title_desc = st.text_input ("Enter title description: ")

    title = number_to_word(title)
    title_desc = number_to_word(title_desc)


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

    print("Cosine similarity score: ", similarity_score) ## nødvendig for nå skal bli slettet før presenteringen av MVP Onsdagen

    avrundet_similarity_score= float("%.3f"%similarity_score)*100
    st.write(similarity_score) ## Samme som kommenter på linje 227
    st.write(avrundet_similarity_score) ## Samme som kommenter 227 og 230
    st.write(avrundet_similarity_score, "%")


    if (avrundet_similarity_score >=70):
        st.write("The title description and title is great")
    elif (avrundet_similarity_score >= 50):
        st.write("The title and description needs a bit more work")
    elif (avrundet_similarity_score >= 30):
        st.write("The title and description needss quite alot of work")
    
    elif (avrundet_similarity_score >= 20):
        st.write("The title and description are quite impossible to comprehend and there is almost none connections")
        
    
    else:
        st.write("It needs a better description")

# Add user icon
st.write("""
    <style>
        .icon {{
            height: 100px;
            width: 100px;
            border-radius: 50%;
            background-color: #333;
            display: inline-block;
        }}
    </style>
    <div class="icon"></div>
""", unsafe_allow_html=True)




st.image("Equinor.jpg", width= 100)
st.title("MVPS")


# Add search bar
search_term = st.text_input("Search")
if search_term:
    st.write("Searching for", search_term)

# Add "Ask a Question" button
if st.button("Ask a Question"):
    st.write("Question asked!")




ch = st.radio(
    "Choose what function to access",
    ('Text Clarity', 'Matching Title', 'Data Sorting'))

if ch=='Text Clarity':
    tex_clar()
elif ch=='Data Sorting':
    dat_sor()
else:
    mat_tit()