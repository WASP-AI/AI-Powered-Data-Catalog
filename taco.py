import streamlit as st
import PyPDF4
import pandas as pd
import re
import os
import mimetypes
import time
from title_desc_comparisoncomp import comparison


# Display the image





def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF4.PdfFileReader(pdf_file)
    text = ""
    for page_number in range(pdf_reader.getNumPages()):
        page = pdf_reader.getPage(page_number)
        text += page.extractText()
    return text
    

def answer_questions(text, file_name):
    question_answers = {
        "File Name": file_name,
        "How often is the data updated?": "N/A",
        "Is additional information defined?": "N/A",
        "Is accuracy defined?": "N/A",
        "Is uniqueness defined?": "N/A",
        "Is timeliness defined?": "N/A",
        "Is information owner defined?": "N/A",
        "Title comparison": "N/A"
    }
    score = 0

    update_pattern = re.compile(r"(daily|weekly|monthly|annually|quarterly|every \d+ (days?|weeks?|months?|years?))", re.IGNORECASE)
    update_match = update_pattern.search(text)
    

    question_answers["Title comparison"]= comparison(uploaded_file)
    if update_match:
        question_answers["How often is the data updated?"] = update_match.group()

    keywords = ["additional information", "accuracy", "uniqueness", "timeliness", "information owner"]
    for keyword in keywords:
        keyword_pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        keyword_match = keyword_pattern.search(text)
        if keyword_match:
            question_answers[f"Is {keyword} defined?"] = "✔️"
            score += 1
#        elif keyword == "Comparison":
 #           question_answers[st.write(comparison(uploaded_file))]
  #          score +=1
        else:
            question_answers[f"Is {keyword} defined?"] = "❌"

    if keyword_match == "Comparison":
        question_answers[f"How accurate is the title compared to the description?"] = comparison(uploaded_file)
        score +=1

    return question_answers, score

st.title("AI Powered Data Catalog")
uploaded_files = st.file_uploader("Choose PDF files", type=["pdf"], accept_multiple_files=True)


if uploaded_files:
    all_answers = []
    total_score = 0
    max_score_per_file = 6
    for uploaded_file in uploaded_files:
        text = extract_text_from_pdf(uploaded_file)
        answers, score = answer_questions(text, uploaded_file.name)
        answers["Score"] = f"{score}/{max_score_per_file}"
        total_score += score

        # Add file size and file type
        file_size = os.path.getsize(uploaded_file.name) / (1024 * 1024)
        file_type, _ = mimetypes.guess_type(uploaded_file.name)
        answers["File Size"] = f"{file_size:.2f} MB"
        answers["File Type"] = file_type

        all_answers.append(answers)

    st.write("Answers:")
    answer_df = pd.DataFrame(all_answers)
    transposed_df = answer_df.T

    st.write(
        transposed_df.style.set_table_styles(
            [
                {"selector": "table", "props": [("width", "150%"), ("table-layout", "auto")]},
                {"selector": "th, td", "props": [("word-wrap", "break-word"), ("white-space", "pre-wrap")]},
            ]
        )
    )
    max_total_score = len(uploaded_files) * max_score_per_file
    st.write(f"Max total Score: {total_score}/{max_total_score}")




