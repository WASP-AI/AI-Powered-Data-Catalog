
# pip install langchain
#('pip install chromadb')
#('pip install unstructured')


import os
os.environ["HUGGINGFACEHUB_API_TOKEN"]= "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"



from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains import VectorDBQA
from langchain.document_loaders import UnstructuredFileLoader
from langchain.prompts import PromptTemplate
from langchain.output_parsers import RegexParser



doc1="""The Arms Shop is a family business owned by Ipponmatsu. 
The shop has been in business for over 200 years. The shop sells antique swords, new swords, and new swords in fashion. 
It also sells clubs, axes, flintlocks, rifles, armor, daggers, spears, etc. 
The shop also takes polishing jobs. 
It used to have a lot of customers until captain Smoker was put in charge of the town. 
Until 2 years ago, the best sword that the shop had was Yubashiri.
Zoro stopped here to get new swords after Mihawk broke two of his swords in their battle. 
Zoro found the cursed sword Sandai Kitetsu, and decided to use it after a quick test. 
Ipponmatsu was amazed at his swordsmanship and gave it to him for free along with Yubashiri, his family heirloom."""

doc2="""He was temporarily forced to join the Foxy Pirates during the Long Ring Long Land Arc, but was quickly returned to Luffy's crew.
Chopper is a reindeer that ate the Hito Hito no Mi, a Devil Fruit that allows its user to transform into a human hybrid or a human at will. 
He came from Drum Island and was taught how to be a doctor by his two parental figures, Doctors Hiriluk and Kureha. 
His dream is to one day become a doctor capable of curing any disease and wants to travel all across the world specifically in the hopes of accomplishing this dream."""

doc3="""In his younger days, he wore an open dark gray vest and a black bandanna, and his trademark mustache was nonexistent. 
Later in his life, he grew the prominent crescent-shaped mustache, which would spawn his epithet, "Whitebeard". 
When he was 52, he wore a white and yellow pirate hat with his jolly roger on it over a black bandana with black and red, both worn over long, blond flowing hair he had back then.
Whitebeard was an abnormally large man, with a height of 666 cm (21'10"). 
Unlike other large-sized humans, however, he was well-proportioned. 
He had a long face, ploughed because of the advanced age with many wrinkles around his eyes, and many scars running along his chest, and was very muscular.
The muscles on his biceps seemed to grow bigger whenever he used his quake-based powers. 
Like all of his men, he had his own Jolly Roger tattooed on his back. 
In the manga, his eyes are brown, but in the anime, his eye color is seen to be yellow."""

mixeddoc= doc1 + "\n\n" + doc2 + "\n\n" + doc3 




loader = UnstructuredFileLoader("Willdelete.txt")
docs= loader.load()





flan_ul2 = HuggingFaceHub(repo_id="google/flan-ul2", model_kwargs={"temprature":0.1, "max_new_tokens":256})




text_splitter = CharacterTextSplitter(chunk_size=150, chunk_overlap=0)
texts = text_splitter.split_text(docs[0].page_content)

embeddings = HuggingFaceHubEmbeddings()
docsearch = Chroma.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])




prompt_template = """Use the following pieces of context to answer the question at the end. If the information can not be found in the context given, just say "dont know".
Don't try to make up an answer. Make sure to only use the context given in {summaries} to answer the question.

{summaries}

Question: {question}
Answer in English:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
)




chain=load_qa_with_sources_chain(flan_ul2, chain_type="stuff", prompt=PROMPT)




question = "Show me a section of the text describing what pirate crews Chopper was a member off."
docs = docsearch.similarity_search(question) 
chain({"input_documents": docs, "question": question})


print(chain({"input_documents": docs, "question": question}))
