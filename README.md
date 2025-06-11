**RAGBOT--2: Modular PDF Retrieval-Augmented Generation Chatbot**
=================================================================

A modular Retrieval-Augmented Generation (RAG) chatbot designed for seamless interaction with your PDF documents. Built using Streamlit, Chroma vector store, and the GROQ LLM, this bot enables efficient retrieval and synthesis of information from uploaded PDFs, all within a user-friendly web interface.

---

**FEATURES**
------------

- **PDF Upload & Parsing:** Easily upload and process PDF documents for knowledge extraction.
- **Chunking & Vectorization:** Documents are split into chunks and embedded for semantic search.
- **Chroma Vector Store:** Fast, in-memory vector search using Chroma for real-time retrieval.
- **GROQ LLM Integration:** Context-aware answer generation using the GROQ large language model.
- **Interactive Streamlit UI:** Clean, intuitive chat interface for engaging with your documents.
- **Modular Codebase:** All logic is organized into reusable modules for clarity and maintainability.

---

**PROJECT STRUCTURE**
---------------------

groq_rag_chatbot/
│
├── index.py # Main Streamlit app
├── requirements.txt # Python dependencies
│
├── moduless/ # All modular logic
│ ├── pdf_handler.py # PDF upload + loading logic
│ ├── vectorstore.py # Chroma in-memory vector store setup
│ ├── llm.py # GROQ LLM and RetrievalQA chain
│ ├── chat.py # Chat interaction logic (input/output)
│ └── chroma_inspector.py # View vector store chunks from sidebar

text

---

**GETTING STARTED**
-------------------

**1. Create a Virtual Environment**
- `python -m venv myenv`
- On Windows: `myenv\Scripts\activate`
- On Mac/Linux: `source myenv/bin/activate`

**2. Install Requirements**
- `pip install -r requirements.txt`

**3. Run the Application**
- `streamlit run index.py`
- The Streamlit UI will launch in your browser for interactive chat.

**4. Upload Your PDFs**
- Use the sidebar in the Streamlit app to upload and process your PDF documents.

---

**WORKFLOW OVERVIEW**
---------------------

- **PDF Handling:**  
  Upload and load PDF files using a dedicated handler module, extracting text for downstream processing.

- **Chunking & Embedding:**  
  Documents are split into manageable chunks and embedded using a semantic embedding model for efficient retrieval.

- **Vector Store Indexing:**  
  Embeddings are stored in a Chroma in-memory vector store, enabling fast similarity search across document chunks.

- **Retrieval:**  
  For each user query, the system retrieves the most relevant chunks from the vector store to provide accurate context.

- **LLM Generation:**  
  Retrieved context and user queries are passed to the GROQ LLM via a RetrievalQA chain, generating detailed, context-aware answers.

- **User Interface:**  
  All interactions occur through an interactive Streamlit chat interface, allowing users to upload documents, ask questions, and view responses seamlessly.

---

**SUMMARY OF KEY TECHNOLOGIES AND MODELS USED**
-----------------------------------------------

- **PDF Handling:** Python, PyPDF (or similar)
- **Chunking & Embedding:** Python, embedding models (as configured)
- **Vector Search:** Chroma (in-memory vector store)
- **LLM Generation:** GROQ LLM, RetrievalQA chain
- **UI:** Streamlit

---

**NOTES**
---------

- Ensure your PDFs are uploaded through the Streamlit interface for proper processing.
- All dependencies are listed in `requirements.txt`.
- For any issues, check the sidebar for vector store inspection and debugging.

---
