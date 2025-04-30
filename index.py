import gradio as gr
import warnings
import logging

# Local modules
from modules.chat import display_chat_history, handle_user_input, download_chat_history
from modules.pdf_handler import upload_pdfs
from modules.vectorstore import load_vectorstore
from modules.llm import get_llm_chain
from modules.chroma_inspector import inspect_chroma

# Silence noisy logs
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# App state
chat_history = []
vectorstore = None

def process_pdfs(files):
    global vectorstore
    if files:
        vectorstore = load_vectorstore(files)
        return "PDFs processed and vectorstore updated."
    return "No files uploaded."

def chatbot_fn(message, history):
    if vectorstore:
        chain = get_llm_chain(vectorstore)
        response = handle_user_input(chain, message, history)
        return response
    else:
        return "Please upload and process PDFs first."

def download_history():
    return download_chat_history(chat_history)

# Custom CSS for background and chat bubble
custom_css = """
body {
    background-image: url('https://t3.ftcdn.net/jpg/05/77/43/10/360_F_577431075_kUXMnnnKgCcvvPVmn66g4yP7mWnsVJRs.jpg');
    background-size: cover;
    background-attachment: fixed;
}
#title-bubble {
    background-color: rgba(0, 0, 0, 0.75);
    padding: 1rem 1.5rem;
    border-radius: 20px;
    margin: 3rem 0 1rem 0;
    text-align: center;
    color: white;
    font-size: 1.4rem;
    box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.4);
}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Column():
        gr.HTML('<div id="title-bubble">How can I help you</div>')

        with gr.Row():
            pdf_input = gr.File(file_types=[".pdf"], file_count="multiple", label="Upload PDFs")
            submit_btn = gr.Button("Process PDFs")

        status = gr.Textbox(label="Status", interactive=False)

        submit_btn.click(fn=process_pdfs, inputs=pdf_input, outputs=status)

        with gr.Accordion("Vectorstore Inspector", open=False):
            gr.Markdown("Placeholder for vectorstore insights")  # You can hook inspect_chroma here

        chatbot = gr.ChatInterface(chatbot_fn)

        with gr.Row():
            gr.Button("Download Chat History").click(download_history, outputs=[])

demo.launch()
