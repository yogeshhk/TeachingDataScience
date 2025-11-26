# Streamlit UI : The user interface allows uploading documents and chatting with the agent.
# It specifically renders retrieved Markdown tables correctly.

import streamlit as st
import os
import tempfile
from backend import OmniIngestor
from agent import rag_agent

st.set_page_config(page_title="Omni-RAG: Multimodal Assistant", layout="wide")

st.title("🧬 Omni-RAG: Multimodal Retrieval System")
st.markdown("Query your complex PDFs, Tables, and Docs with **Groq + Docling + LangGraph**")

# Sidebar for Setup
with st.sidebar:
    st.header("1. Ingestion")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if st.button("Process Document"):
        if uploaded_file and os.getenv("GROQ_API_KEY"):
            with st.spinner("Parsing Structure with Docling..."):
                # Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                # Ingest
                ingestor = OmniIngestor()
                chunks = ingestor.process_pdf(tmp_path)
                st.success(f"Ingested {len(chunks)} semantic chunks!")
        else:
            st.error("Please upload a file and set API Key.")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about specific tables or data in your doc..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Agent is reasoning..."):
            initial_state = {"question": prompt, "retry_count": 0}
            result = rag_agent.invoke(initial_state)
            response = result["answer"]
            
            st.markdown(response)
            
            # Show sources in an expander
            with st.expander("View Source Context"):
                for ctx in result["context"]:
                    st.markdown("---")
                    st.markdown(ctx) # Renders Markdown tables beautifully
    
    st.session_state.messages.append({"role": "assistant", "content": response})