"""
Bajaj Finserv Mutual Fund Chatbot - Main Entry Point
Interactive Q&A interface for mutual fund factsheet queries
"""
import streamlit as st
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from rag.langgraph_workflow import RAGWorkflow

# Page config
st.set_page_config(
    page_title="Bajaj Finserv Fund Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 0.25rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .confidence-high {
        color: #4caf50;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff9800;
        font-weight: bold;
    }
    .confidence-low {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_rag():
    """Initialize RAG workflow (cached to avoid reloading)"""
    with st.spinner("🚀 Initializing AI system..."):
        return RAGWorkflow(vector_storage_type='faiss')


def format_confidence(confidence):
    """Format confidence with color"""
    if confidence == "High":
        return f'<span class="confidence-high">● {confidence}</span>'
    elif confidence == "Medium":
        return f'<span class="confidence-medium">● {confidence}</span>'
    else:
        return f'<span class="confidence-low">● {confidence}</span>'


def main():
    # Header
    st.markdown('<div class="main-header">💼 Bajaj Finserv Fund Chatbot</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Ask questions about mutual fund factsheets</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📊 About")
        st.markdown("""
        This AI-powered chatbot helps you explore Bajaj Finserv mutual fund factsheets.
        
        **Features:**
        - 💬 Natural language Q&A
        - 📈 Fund comparisons
        - 🔍 Smart semantic search
        - 📑 Source citations
        
        **Coverage:**
        - 19 mutual funds
        - 201 knowledge chunks
        - 56 pages analyzed
        """)
        
        st.divider()
        
        st.header("💡 Example Questions")
        example_queries = [
            "What is the AUM of Large Cap Fund?",
            "Which fund has highest 1-year return?",
            "Show top 5 holdings of Flexi Cap Fund",
            "Compare expense ratios of all funds",
            "What is the risk profile of Small Cap Fund?",
            "Investment philosophy of Large Cap Fund?"
        ]
        
        for query in example_queries:
            if st.button(query, key=query, use_container_width=True):
                st.session_state.selected_query = query
        
        st.divider()
        
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Initialize RAG
    if "rag" not in st.session_state:
        st.session_state.rag = initialize_rag()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
            
            if message["role"] == "assistant" and "metadata" in message:
                # Show metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.caption(f"📄 Chunks: {message['metadata']['chunks']}")
                with col2:
                    st.caption(f"🎯 Type: {message['metadata']['query_type']}")
                with col3:
                    confidence_html = format_confidence(message['metadata']['confidence'])
                    st.markdown(f"**Confidence:** {confidence_html}", unsafe_allow_html=True)
                
                # Show sources
                if message['metadata'].get('sources'):
                    st.caption("📚 **Sources:**")
                    sources_html = " ".join([f'<span class="source-badge">{s}</span>' for s in message['metadata']['sources']])
                    st.markdown(sources_html, unsafe_allow_html=True)
    
    # Handle example query selection
    if "selected_query" in st.session_state:
        query = st.session_state.selected_query
        del st.session_state.selected_query
    else:
        # Chat input
        query = st.chat_input("Ask a question about Bajaj Finserv mutual funds...")
    
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.rag.query(query)
                    
                    # Display answer
                    st.markdown(result['answer'])
                    
                    # Display metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"📄 Chunks: {result['num_chunks']}")
                    with col2:
                        st.caption(f"🎯 Type: {result['query_type']}")
                    with col3:
                        confidence_html = format_confidence(result['confidence'])
                        st.markdown(f"**Confidence:** {confidence_html}", unsafe_allow_html=True)
                    
                    # Display sources
                    if result.get('sources'):
                        st.caption("📚 **Sources:**")
                        sources_html = " ".join([f'<span class="source-badge">{s}</span>' for s in result['sources']])
                        st.markdown(sources_html, unsafe_allow_html=True)
                    
                    # Add to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result['answer'],
                        "metadata": {
                            "chunks": result['num_chunks'],
                            "query_type": result['query_type'],
                            "confidence": result['confidence'],
                            "sources": result.get('sources', [])
                        }
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })
    
    # Footer
    st.divider()
    st.caption("🤖 Powered by LLM | 🔍 FAISS Vector Search | 📊 LangGraph RAG Pipeline")


if __name__ == "__main__":
    main()
