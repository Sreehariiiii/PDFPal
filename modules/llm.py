import os
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()  # Loads variables from a .env file into os.environ

def get_llm_chain(vectorstore):
    # Get API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables.")

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
