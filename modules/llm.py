import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env (if needed)
load_dotenv()

def get_llm_chain(vectorstore):
    # Ensure the API key is in your Streamlit secrets
    groq_api_key = st.secrets["groq"]["GROQ_API_KEY"]

    # Initialize the LLM with the API key and model name
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192"  # Make sure this model exists in Hugging Face
    )

    # Set up the retriever for the vector store
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Return the retrieval chain with LLM, chain type, and retriever
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
