import gradio as gr

# Global state
chat_history = []

def display_chat_history():
    # Display previous chat messages in Gradio
    return [(msg["role"], msg["content"]) for msg in chat_history]

def handle_user_input(chain, user_input):
    global chat_history

    # User message
    chat_history.append({"role": "user", "content": user_input})

    # Get the model response
    try:
        result = chain({"query": user_input})
        response = result["result"]
        chat_history.append({"role": "assistant", "content": response})

        return display_chat_history()  # Update the chat history UI

    except Exception as e:
        return f"Error: {str(e)}"

def download_chat_history():
    if chat_history:
        content = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in chat_history])
        return content
    return "No chat history available."

# Gradio UI
with gr.Blocks() as demo:
    chatbot = gr.Chatbot(value=display_chat_history)
    user_input = gr.Textbox(placeholder="Type your message here...", label="User Input")
    submit_btn = gr.Button("Submit")
    download_btn = gr.Button("Download Chat History")

    # Define actions for buttons
    submit_btn.click(handle_user_input, inputs=[gr.State(), user_input], outputs=chatbot)
    download_btn.click(download_chat_history, outputs=gr.File())

demo.launch()
