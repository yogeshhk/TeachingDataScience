# 📚 Book QA System - Streamlit UI

A Streamlit-based web application for asking questions about PDF books using Docling for document processing and LangChain for question-answering.

## Features

- **PDF Upload**: Easy file upload through the sidebar
- **Document Processing**: Uses Docling for advanced PDF processing with OCR and table structure recognition
- **Vector Search**: FAISS-based vector search for relevant document chunks
- **Interactive Chat**: Chat interface for asking questions about your book
- **Source Documents**: View the source chunks used to answer each question
- **Document Preview**: See the raw Docling markdown output in the right panel
- **Persistent Storage**: Vector stores are saved and reused for faster subsequent loads

## Setup Instructions

### 1. Install Dependencies

**Method A: Using Setup Script (Recommended)**
```bash
python setup.py
```

**Method B: Manual Installation**
```bash
# First, install specific versions to avoid conflicts
pip install numpy==1.24.3 pandas==2.1.4 bottleneck==1.3.7

# Then install remaining dependencies
pip install -r requirements.txt
```

**Method C: Alternative Requirements File**
```bash
pip install -r requirements_alternative.txt
```

### 2. Set up Local LLM (Required)

This application uses a local LLM server. You need to set up a local model server at `http://localhost:1234/v1`.

**Option A: Using LM Studio**
1. Download and install [LM Studio](https://lmstudio.ai/)
2. Load your preferred model
3. Start the local server on port 1234

**Option B: Using Ollama**
1. Install [Ollama](https://ollama.ai/)
2. Pull a model: `ollama pull llama2`
3. Start the server: `ollama serve`
4. Update the API endpoint in `doclingbookloader.py` if needed

### 3. Run the Application

```bash
streamlit run streamlit_app.py
```

## Usage

1. **Upload PDF**: Use the sidebar to upload a PDF book
2. **Processing**: Wait for the document to be processed by Docling
3. **Ask Questions**: Use the chat interface to ask questions about your book
4. **View Sources**: Expand the "Source Documents" section to see relevant chunks
5. **Preview Content**: Check the right panel to see the extracted markdown content

## File Structure

```
├── streamlit_app.py          # Main Streamlit application
├── doclingbookloader.py      # Document loader and QA system classes
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Configuration

### Docling Pipeline Options

The application uses the following Docling configuration:
- **OCR**: Enabled for text extraction from images
- **Table Structure**: Enabled with cell matching
- **Accelerator**: Auto-detection with 8 threads
- **Chunk Size**: 1000 characters with 200 character overlap

### LLM Configuration

Current settings in `doclingbookloader.py`:
- **Model**: `local-model`
- **API Base**: `http://localhost:1234/v1`
- **Temperature**: 0 (deterministic responses)
- **Search Type**: MMR (Maximum Marginal Relevance)
- **Retrieved Chunks**: 5 per query

## Troubleshooting

### Common Issues

1. **"AttributeError: ARRAY_API not found" (Bottleneck Error)**:
   ```bash
   # Solution 1: Use the setup script
   python setup.py
   
   # Solution 2: Manual fix
   pip uninstall bottleneck pandas numpy -y
   pip install numpy==1.24.3 pandas==2.1.4 bottleneck==1.3.7
   pip install -r requirements.txt
   
   # Solution 3: Alternative approach
   pip install -r requirements_alternative.txt
   ```

2. **"Connection Error"**: Ensure your local LLM server is running
3. **"FAISS Import Error"**: Install with `pip install faiss-cpu`
4. **"Tokenizer Warnings"**: These are normal and don't affect functionality
5. **"Memory Issues"**: Try reducing chunk size or using smaller models

### Environment Issues

If you're having persistent dependency conflicts:

```bash
# Create a fresh conda environment
conda create -n bookqa python=3.10
conda activate bookqa

# Then run the setup
python setup.py
```

### Performance Tips

- **First Load**: Initial processing takes time but creates a reusable index
- **Subsequent Loads**: Vector stores are cached for faster startup
- **Large Files**: Consider splitting very large PDFs for better performance

## Customization

### Change LLM Provider

To use a different LLM provider, modify the `ChatOpenAI` initialization in `doclingbookloader.py`:

```python
# For OpenAI
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    openai_api_key="your-api-key",
    temperature=0,
)

# For Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(
    model="claude-3-sonnet-20240229",
    anthropic_api_key="your-api-key",
    temperature=0,
)
```

### Adjust Chunk Settings

Modify the text splitter parameters in `doclingbookloader.py`:

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,        # Increase for more context
    chunk_overlap=300,      # Increase for better continuity
    separators=["\n\n", "\n", " ", ""]
)
```

## License

This project is open source and available under the MIT License.
