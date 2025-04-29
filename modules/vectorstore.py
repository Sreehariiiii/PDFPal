import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from modules.pdf_handler import save_uploaded_files
import os

PERSIST_DIR = "./chroma_store"

def load_vectorstore(uploaded_files):
    # Fetch Hugging Face API token from Streamlit secrets
    hf_token = st.secrets["huggingface"]["HUGGINGFACEHUB_API_TOKEN"]

    # Set the environment variable for Hugging Face API token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token

    # Save uploaded files
    paths = save_uploaded_files(uploaded_files)

    docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    # Initialize embeddings (token is automatically picked from environment variable)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # Append to existing vectorstore
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        vectorstore.add_documents(texts)
        vectorstore.persist()
    else:
        # Create a new vectorstore
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()

    return vectorstore
