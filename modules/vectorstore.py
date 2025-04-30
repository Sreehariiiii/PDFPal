import os
import pickle
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

PERSIST_DIR = "./faiss_store"

def save_uploaded_files(uploaded_files):
    # Create a directory to save files if not already exists
    if not os.path.exists("uploaded_files"):
        os.makedirs("uploaded_files")
    
    file_paths = []
    for file in uploaded_files:
        file_path = os.path.join("uploaded_files", file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_paths.append(file_path)
    
    return file_paths

def load_vectorstore(uploaded_files):
    paths = save_uploaded_files(uploaded_files)

    docs = []
    for path in paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L12-v2")

    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        # Append to existing FAISS
        with open(PERSIST_DIR + "/faiss_index.pkl", 'rb') as f:
            vectorstore = pickle.load(f, allow_dangerous_deserialization=True)  # Add the deserialization flag here
        vectorstore.add_documents(texts)
        vectorstore.save_local(PERSIST_DIR)
    else:
        # Create new FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=texts,
            embedding=embeddings
        )
        vectorstore.save_local(PERSIST_DIR)

    return vectorstore
