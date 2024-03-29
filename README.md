# Bachelor Thesis with Equinor

Introducing our awesome Equinor Data Catalog Quality Assurance App! 

🎉 This app aims to ensure that all data catalog entries on Equinor's data platform are accurate, complete, and meet a certain standard. We know that inconsistent documentation can lead to trust issues and extra work for data product users, and we're here to help! Our app will not only quality-check existing documentation, but also assist creators in maintaining a high standard for new entries. Built with user stories in mind, this app empowers data product owners and stewards to create and maintain top-notch documentation with ease. 

Let's make the Equinor Data Catalog better together! 🚀


# Task solution

We are working on developing an NLP web application using LLMs. The web application runs on the Streamlit library. It uses the google FLAN-UL2 model to quality ensure data catalog entries from the user. 

Equinor has provided us with a best practice document, with information about what a data product entry should include to be Top-Notch!.📖 
The document includes a checklist, going through each requirement 📋.

The program, with the help of the best practice document, will follow these requirements to solve the issue that the user hands it. 

User will upload the document. The program will then output a table, with the requirements of the best practice document, and display which requirements are met, and the areas that still needs work to be complete ✅



# Libraries

The program uses the following python libraries:

Langchain: Mainly used with the LLM (FLAN-UL2). Parameters and prompts can be modified using Langchain functions. Download the library and get started with Langchain here: https://python.langchain.com/en/latest/getting_started/getting_started.html

Streamlit: Used in the development of the Web Application. Streamlit provides a set of functions that supports python programmers in front-end development where one can track changes in the web app while simultaneously modifying the code in Python. Get started: https://streamlit.io/

Huggingface: To access the FLAN-UL2 model, you must use a HuggingFace API token. Also, the program uses HuggingFace embeddings for embedding the input.
To access your Huggingface API token, click "New token" on the Access tokens section of Settings: https://huggingface.co/settings/tokens
