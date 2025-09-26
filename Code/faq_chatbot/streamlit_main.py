"""
Streamlit UI for FAQ Chatbot
This module provides a web interface for the RAG-based FAQ chatbot
with CSV upload functionality and chat interaction.
"""

import streamlit as st
import pandas as pd
import tempfile
import os
from main_faq_chatbot import FAQChatbot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="FAQ Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'faq_loaded' not in st.session_state:
        st.session_state.faq_loaded = False

def display_chat_message(message, is_user=True):
    """Display a chat message with appropriate styling."""
    if is_user:
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)

def load_faq_from_upload(uploaded_file, similarity_threshold):
    """Load FAQ data from uploaded CSV file."""
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Validate CSV format
        df = pd.read_csv(tmp_file_path)
        if len(df.columns) < 2:
            st.error("CSV file must have at least 2 columns (questions and answers)")
            os.unlink(tmp_file_path)
            return None
        
        # Display preview
        st.sidebar.success(f"✅ CSV loaded successfully! ({len(df)} FAQ pairs)")
        with st.sidebar.expander("Preview FAQ Data"):
            st.dataframe(df.head())
        
        # Initialize chatbot
        chatbot = FAQChatbot(tmp_file_path, similarity_threshold)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        return chatbot
        
    except Exception as e:
        logger.error(f"Error loading FAQ from upload: {e}")
        st.error(f"Error loading CSV: {str(e)}")
        if 'tmp_file_path' in locals():
            os.unlink(tmp_file_path)
        return None

def main():
    """Main Streamlit application."""
    
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🤖 FAQ Chatbot")
    st.markdown("Upload your FAQ CSV file and start chatting!")
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload FAQ CSV File",
            type=['csv'],
            help="CSV should have two columns: questions and answers"
        )
        
        # Similarity threshold
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Minimum similarity score to return an answer (higher = more strict)"
        )
        
        # API Key status
        groq_api_key = os.getenv('GROQ_API_KEY')
        if groq_api_key:
            st.success("✅ GROQ API Key found")
        else:
            st.error("❌ GROQ API Key not found")
            st.info("Please set GROQ_API_KEY environment variable")
        
        # Load chatbot when file is uploaded
        if uploaded_file is not None and groq_api_key:
            if not st.session_state.faq_loaded:
                with st.spinner("Loading FAQ data..."):
                    st.session_state.chatbot = load_faq_from_upload(
                        uploaded_file, 
                        similarity_threshold
                    )
                    if st.session_state.chatbot:
                        st.session_state.faq_loaded = True
            
            # Display FAQ statistics
            if st.session_state.chatbot:
                st.subheader("📊 FAQ Statistics")
                stats = st.session_state.chatbot.get_faq_stats()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total FAQs", stats.get('total_faqs', 'N/A'))
                with col2:
                    st.metric("Threshold", f"{stats.get('similarity_threshold', 'N/A'):.1f}")
        
        # Reset button
        if st.button("🔄 Reset Chat"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Clear FAQ button
        if st.session_state.faq_loaded and st.button("🗑️ Clear FAQ Data"):
            st.session_state.chatbot = None
            st.session_state.faq_loaded = False
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    if not groq_api_key:
        st.warning("⚠️ Please set your GROQ API key as an environment variable to use the chatbot.")
        st.info("Set GROQ_API_KEY in your environment or deployment platform")
        return
    
    if not st.session_state.faq_loaded:
        st.info("👆 Please upload a FAQ CSV file in the sidebar to start chatting!")
        
        # Show sample format
        st.subheader("📋 Expected CSV Format")
        sample_df = pd.DataFrame({
            'question': [
                'What is your return policy?',
                'How long does shipping take?',
                'Do you offer customer support?'
            ],
            'answer': [
                'We offer a 30-day return policy for all unused items.',
                'Standard shipping takes 5-7 business days.',
                'Yes, we offer 24/7 customer support via email and chat.'
            ]
        })
        st.dataframe(sample_df)
        return
    
    # Chat interface
    st.subheader("💬 Chat with FAQ Bot")
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(message['content'], message['is_user'])
    
    # Chat input
    if user_input := st.chat_input("Ask me anything about the FAQ..."):
        # Add user message to history
        st.session_state.chat_history.append({
            'content': user_input,
            'is_user': True
        })
        display_chat_message(user_input, is_user=True)
        
        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.chatbot.query(user_input)
                    st.write(response)
                    
                    # Add bot response to history
                    st.session_state.chat_history.append({
                        'content': response,
                        'is_user': False
                    })
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({
                        'content': error_msg,
                        'is_user': False
                    })
    
    # Help section
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **Steps to get started:**
        1. Set your `GROQ_API_KEY` environment variable
        2. Upload a CSV file with FAQ data (questions in first column, answers in second)
        3. Adjust the similarity threshold if needed
        4. Start asking questions!
        
        **Tips:**
        - Higher similarity threshold = more strict matching
        - Lower similarity threshold = more flexible matching
        - The bot will find the most similar FAQ question and return its answer
        """)

if __name__ == "__main__":
    main()