import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

def get_llm_chain(vectorstore):
    groq_api_key = st.secrets["groq"]["GROQ_API_KEY"],
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
