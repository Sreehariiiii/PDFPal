import streamlit as st

def display_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            # User message: dark background, white text
            st.markdown(f"""
                <div style="background-color: rgba(0, 0, 0, 0.85); color: white; padding: 0.75rem 1rem; border-radius: 10px; margin: 0.5rem 0; width: fit-content; max-width: 80%;">
                    {content}
                </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message: light background, black text
            st.markdown(f"""
                <div style="background-color: rgba(255, 255, 255, 0.85); color: black; padding: 0.75rem 1rem; border-radius: 10px; margin: 0.5rem 0; width: fit-content; max-width: 80%;">
                    {content}
                </div>
            """, unsafe_allow_html=True)

def handle_user_input(chain):
    user_input = st.chat_input("Pass your prompt here")
    if not user_input:
        return
    
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    try:
        result = chain({"query": user_input})
        response = result["result"]
        st.markdown(f"""
    <div style="background-color: rgba(255, 255, 255, 0.85); color: black; padding: 0.75rem 1rem; border-radius: 10px; margin: 0.5rem 0; width: fit-content; max-width: 80%;">
        {response}
    </div>
""", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": response})
    except Exception as e:
        st.error(f"Error: {str(e)}")

def download_chat_history():
    if st.session_state.get("messages"):
        content = "\n\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("💾 Download Chat History", content, file_name="chat_history.txt", mime="text/plain")
