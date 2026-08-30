import streamlit as st
import os
from llm_llamaindex_rag import ResumeRAG

# --- Page Configuration ---
st.set_page_config(
    page_title="Resume AI Assistant",
    page_icon="📄",
    layout="wide"
)

# --- State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

# --- Main UI ---
st.title("📄 AI-Powered Resume Assistant")
st.markdown("Upload resumes and ask questions to get instant insights.")

# --- API Key Input ---
groq_api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# --- File Uploader and RAG Initialization ---
with st.sidebar:
    st.header("Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload your resume files (txt, pdf, docx)", 
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx']
    )
    
    if st.button("Build RAG System"):
        if uploaded_files and groq_api_key:
            data_dir = "uploaded_resumes"
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            
            for file in uploaded_files:
                with open(os.path.join(data_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())
            
            with st.spinner("Building RAG system... This may take a moment."):
                st.session_state.rag_system = ResumeRAG(groq_api_key=groq_api_key, data_dir=data_dir)
            st.success("RAG system built successfully!")
        elif not groq_api_key:
            st.warning("Please enter your Groq API key.")
        else:
            st.warning("Please upload at least one resume file.")

# --- Chat Interface ---
st.header("Chat with Your Resumes")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the resumes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if st.session_state.rag_system:
            with st.spinner("Thinking..."):
                response = st.session_state.rag_system.query(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
        else:
            st.warning("Please upload resumes and build the RAG system first.")
            st.session_state.messages.append({"role": "assistant", "content": "Please upload resumes and build the RAG system first."})