from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma # importing Chroma vector store
from langchain_core.documents import Document
import os
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "realistic_restaurant_reviews.csv")
df = pd.read_csv(csv_path) # loading the dataset
embeddings = OllamaEmbeddings(model="mxbai-embed-large") # loading the embedding model

db_location = os.path.join(script_dir, "chroma_langchain_db") # location to store the vector database
add_documents = not os.path.exists(db_location) # checking if the DB already exists

if add_documents: # if it doesn't exist, we need to add the documents from the dataframe we made
    documents = []
    ids = []
    for i, row in df.iterrows(): # iterating through the dataframe rows
        document = Document(
            page_content = row["Title"] + "" + row["Review"], # format of the data in the document we're making
            metadata = {"rating": row["Rating"], "date": row["Date"]},
            id = str(i) # unique id for each document
        )
        ids.append(str(i)) # adding the id to the list of ids
        documents.append(document)

vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids) 
    # if it doesn't exist already, add the documents to the vector store, passing the related docs and ids

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5} # setting the number of documents to retrieve
    )
    