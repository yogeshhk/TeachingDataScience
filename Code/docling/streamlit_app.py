import streamlit as st
import os
import tempfile
import time
from docling_book_loader import DoclingBookLoader, BookQASystem

# Configure the Streamlit page
st.set_page_config(
    page_title="📚 Book QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'qa_system' not in st.session_state:
    st.session_state.qa_system = BookQASystem()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'current_book' not in st.session_state:
    st.session_state.current_book = None
if 'docling_content' not in st.session_state:
    st.session_state.docling_content = None

# Sidebar for file upload
with st.sidebar:
    st.title("📁 File Upload")
    st.markdown("Upload a PDF book to start asking questions!")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF book to create a QA system"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # Check if it's a new file
        if st.session_state.current_book != uploaded_file.name:
            st.session_state.current_book = uploaded_file.name
            st.session_state.pdf_processed = False
            st.session_state.messages = []
            st.session_state.qa_system.reset_chat_history()
            st.session_state.docling_content = None
        
        # Process the PDF
        if not st.session_state.pdf_processed:
            st.info("📚 Processing your book...")
            
            # Create progress container
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(message):
                    status_text.text(message)
                    progress_bar.progress(min(progress_bar.progress + 0.2, 0.9))
                
                try:
                    # Create vector store
                    st.session_state.qa_system.create_vectorstore(
                        tmp_file_path, 
                        progress_callback=update_progress
                    )
                    
                    # Create QA chain
                    status_text.text("Creating QA chain...")
                    progress_bar.progress(0.95)
                    st.session_state.qa_system.create_qa_chain(uploaded_file.name)
                    
                    # Get docling content for display
                    status_text.text("Getting document content...")
                    loader = DoclingBookLoader(tmp_file_path)
                    docling_doc = loader.get_docling_document()
                    st.session_state.docling_content = docling_doc.export_to_markdown()
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Book processed successfully!")
                    st.session_state.pdf_processed = True
                    
                    # Clean up temporary file
                    os.unlink(tmp_file_path)
                    
                    st.success("🎉 Your book is ready for questions!")
                    time.sleep(2)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing book: {str(e)}")
                    os.unlink(tmp_file_path)
    
    # Book information
    if st.session_state.current_book:
        st.markdown("---")
        st.markdown("### 📖 Current Book")
        st.info(f"**{st.session_state.current_book}**")
        
        if st.session_state.pdf_processed:
            st.success("✅ Ready for questions")
            
            # Reset chat button
            if st.button("🔄 Reset Chat", help="Clear chat history"):
                st.session_state.messages = []
                st.session_state.qa_system.reset_chat_history()
                st.rerun()
        else:
            st.warning("⏳ Processing...")

# Main content area
col1, col2 = st.columns([3, 2])

with col1:
    # Title
    st.title("📚 Book QA System")
    st.markdown("Ask questions about your uploaded PDF book!")
    
    # Chat interface
    if st.session_state.pdf_processed:
        st.markdown("---")
        st.subheader("💬 Chat with your book")
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show source documents for assistant responses
                if message["role"] == "assistant" and "sources" in message:
                    with st.expander("📄 Source Documents"):
                        for i, doc in enumerate(message["sources"], 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                            st.markdown("---")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your book..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = st.session_state.qa_system.ask_question(
                            prompt, 
                            st.session_state.current_book
                        )
                        
                        response = result["answer"]
                        sources = result["source_documents"]
                        
                        st.markdown(response)
                        
                        # Add assistant message with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": response,
                            "sources": sources
                        })
                        
                        # Show source documents
                        with st.expander("📄 Source Documents"):
                            for i, doc in enumerate(sources, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.markdown("---")
                        
                    except Exception as e:
                        error_message = f"❌ Error: {str(e)}"
                        st.error(error_message)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_message
                        })
    
    else:
        st.info("👈 Please upload a PDF file to start asking questions!")

with col2:
    if st.session_state.docling_content:
        st.subheader("📄 Document Content")
        st.markdown("*Extracted content in Docling's internal format:*")
        
        # Display the docling content in a scrollable text area
        st.text_area(
            label="Docling Markdown Output",
            value=st.session_state.docling_content,
            height=600,
            disabled=True,
            help="This shows the raw markdown content extracted by Docling"
        )
        
        # Download button for the content
        st.download_button(
            label="💾 Download Markdown",
            data=st.session_state.docling_content,
            file_name=f"{st.session_state.current_book}_docling_output.md",
            mime="text/markdown",
            help="Download the extracted content as a markdown file"
        )
    
    else:
        st.info("📄 Document content will appear here after processing")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Built with ❤️ using Streamlit, Docling, and LangChain
    </div>
    """,
    unsafe_allow_html=True
)
