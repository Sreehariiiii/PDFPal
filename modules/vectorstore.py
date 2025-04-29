from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import os

PERSIST_DIR = "./chroma_store"

def load_vectorstore(uploaded_files):
    paths = save_uploaded_files(uploaded_files)

    docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    # Initialize text splitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    # Get Hugging Face token from secrets
    hf_token = st.secrets["huggingface"]["HUGGINGFACEHUB_API_TOKEN"]  # Access the HuggingFace token from secrets

    # Initialize embeddings with Hugging Face token
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L12-v2", 
        huggingfacehub_api_token=hf_token  # Pass the token here
    )

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # Append to existing
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        vectorstore.add_documents(texts)
        vectorstore.persist()
    else:
        # Create new
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()

    return vectorstore
