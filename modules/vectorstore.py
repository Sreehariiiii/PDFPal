from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from modules.pdf_handler import save_uploaded_files
from huggingface_hub import login
import streamlit as st
import os

PERSIST_DIR = "./chroma_store"

def load_vectorstore(uploaded_files):
    paths = save_uploaded_files(uploaded_files)

    docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    # ✅ Login to Hugging Face with token
    hf_token = st.secrets["huggingface"]["HUGGINGFACEHUB_API_TOKEN"]
    login(token=hf_token)

    # ✅ Initialize embeddings without passing token here (it uses the authenticated session)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    # ✅ Chroma vector store creation
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
        vectorstore.add_documents(texts)
        vectorstore.persist()
    else:
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIR
        )
        vectorstore.persist()

    return vectorstore
