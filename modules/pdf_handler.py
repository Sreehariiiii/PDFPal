import gradio as gr
import tempfile

def upload_pdfs():
    with gr.Column():
        gr.Markdown("📁 Upload PDFs")
        uploaded_files = gr.File(file_types=["pdf"], file_count="multiple", label="Choose PDF files")
        submit = gr.Button("Submit to DB")
    
    return uploaded_files, submit

def save_uploaded_files(uploaded_files):
    file_paths = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            file_paths.append(tmp.name)
    return file_paths
