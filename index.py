import warnings
import logging
import streamlit as st

# Local modules
from modules.chat import display_chat_history, handle_user_input, download_chat_history
from modules.pdf_handler import upload_pdfs
from modules.vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.chroma_inspector import inspect_chroma

# Silence noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

st.set_page_config(
    page_title="RagBot!",     
)
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://t3.ftcdn.net/jpg/05/77/43/10/360_F_577431075_kUXMnnnKgCcvvPVmn66g4yP7mWnsVJRs.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .custom-bubble {
        background-color: rgba(0, 0, 0, 0.75);
        padding: 1rem 1.5rem;
        border-radius: 20px;
        margin: 3rem 0 1rem 0;
        display: flex;
        justify-content: center;
        align-items: center;
        width: fit-content;
        height: 60px;
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.4);
    }

    .custom-bubble h1 {
        color: white;
        font-size: 1.4rem;
        margin: 0;
        text-align: center;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add this for the title bubble:
st.markdown('<div class="custom-bubble"><h1>How can I help you</h1></div>', unsafe_allow_html=True)



# Step 1: Upload PDFs + wait for submit
uploaded_files, submitted = upload_pdfs()

# Step 2: If user clicks submit, update vectorstore
if submitted and uploaded_files:
    with st.spinner(" Updating vector database..."):
        vectorstore = load_vectorstore(uploaded_files)
        st.session_state.vectorstore = vectorstore


# Step 3: Display vectorstore inspector (Sidebar)
if "vectorstore" in st.session_state:
    inspect_chroma(st.session_state.vectorstore)

# Step 4: Display old chat messages
display_chat_history()

# Step 5: Handle new prompt input
if "vectorstore" in st.session_state:
    handle_user_input(get_llm_chain(st.session_state.vectorstore))

# Step 6: Chat history export
download_chat_history()