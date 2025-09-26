# FAQ Chatbot with RAG (Retrieval Augmented Generation)

A sophisticated FAQ chatbot built using Retrieval Augmented Generation (RAG) approach with LlamaIndex and HuggingFace LLM API. The system vectorizes FAQ questions, performs semantic similarity search, and returns relevant answers from your FAQ dataset.

## 🌟 Features

- **RAG-based Question Answering**: Uses vector similarity to match user queries with FAQ questions
- **Streamlit Web Interface**: User-friendly web app with CSV upload capability
- **Flexible CSV Support**: Works with any CSV containing question-answer pairs
- **Configurable Similarity Threshold**: Adjust matching strictness
- **Comprehensive Benchmarking**: Evaluate performance using cosine similarity or LLM judge
- **Real-time Statistics**: View FAQ dataset statistics and chatbot performance metrics

## 📁 Project Structure

```
faq-chatbot/
├── main_faq_chatbot.py     # Core chatbot implementation with RAG
├── streamlit_main.py       # Streamlit web interface
├── benchmark_testing.py    # Performance evaluation system
├── README.md              # This file
├── requirements.txt       # Python dependencies
└── sample_faq.csv         # Sample FAQ data (auto-generated)
```

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed on your system
2. **Groq API Key** - Get one from [Groq Console](https://console.groq.com/)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up your Groq API key**:
```bash
# On Linux/Mac
export GROQ_API_KEY="your_groq_api_key_here"

# On Windows
set GROQ_API_KEY=your_groq_api_key_here
```

### Running the Application

#### Option 1: Streamlit Web Interface (Recommended)
```bash
streamlit run streamlit_main.py
```
- Open your browser to the provided URL (usually http://localhost:8501)
- Upload your FAQ CSV file using the sidebar
- Start chatting with your FAQ bot!

#### Option 2: Command Line Testing
```bash
python main_faq_chatbot.py
```
This will create sample FAQ data and test the chatbot with example questions.

#### Option 3: Benchmark Testing
```bash
# Test with cosine similarity (default)
python benchmark_testing.py --csv_file your_faq.csv --questions 10

# Test with LLM judge evaluation
python benchmark_testing.py --csv_file your_faq.csv --mode llm --questions 5
```

## 📋 CSV Format

Your FAQ CSV file should have at least 2 columns:

| question | answer |
|----------|--------|
| What is your return policy? | We offer a 30-day return policy for all unused items. |
| How long does shipping take? | Standard shipping takes 5-7 business days. |
| Do you offer customer support? | Yes, we offer 24/7 customer support via email and chat. |

**Notes:**
- First column: Questions that users might ask
- Second column: Corresponding answers
- Additional columns are ignored
- Empty rows are automatically filtered out

## 🔧 Configuration Options

### Similarity Threshold
- **Range**: 0.1 to 1.0
- **Default**: 0.7
- **Lower values**: More flexible matching, may return less relevant answers
- **Higher values**: Stricter matching, may miss relevant questions

### Evaluation Modes (Benchmark Testing)
- **`cosine`**: Uses sentence embeddings and cosine similarity
- **`llm`**: Uses LLM as a judge to evaluate answer quality

## 📊 File Descriptions

### `main_faq_chatbot.py`
The core chatbot implementation featuring:
- **FAQChatbot Class**: Main chatbot logic with RAG implementation
- **Vector Indexing**: Creates embeddings for FAQ questions using HuggingFace models
- **Similarity Search**: Matches user queries to FAQ questions using vector similarity
- **Metadata Storage**: Stores answers as metadata with question vectors
- **Groq LLM Integration**: Uses Groq API for language model capabilities

**Key Methods:**
- `query(user_question)`: Main method to get answers for user questions
- `get_faq_stats()`: Returns statistics about loaded FAQ data

### `streamlit_main.py`
Web interface built with Streamlit:
- **File Upload**: Drag-and-drop CSV upload with validation
- **Chat Interface**: Real-time chat with conversation history
- **Configuration Panel**: Adjust similarity threshold and view statistics
- **Error Handling**: User-friendly error messages and status indicators

**Features:**
- Session state management for chat history
- Real-time FAQ statistics display
- CSV format validation and preview
- Responsive design with sidebar configuration

### `benchmark_testing.py`
Comprehensive testing framework:
- **Random Sampling**: Selects random questions from FAQ dataset
- **Dual Evaluation**: Cosine similarity and LLM judge methods
- **Performance Metrics**: Score distribution, response times, accuracy
- **Detailed Reporting**: Question-by-question analysis and CSV export

**Evaluation Methods:**
1. **Cosine Similarity**: Compares semantic similarity of generated vs expected answers
2. **LLM Judge**: Uses language model to evaluate answer quality (0.0-1.0 scale)

## 🛠️ Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
llama-index>=0.9.0
llama-index-llms-groq>=0.1.0
llama-index-embeddings-huggingface>=0.1.0
sentence-transformers>=2.2.0
scikit-learn>=1.3.0
numpy>=1.24.0
```

## 💡 Usage Examples

### Basic Query
```python
from main_faq_chatbot import FAQChatbot

# Initialize chatbot
chatbot = FAQChatbot('your_faq.csv', similarity_threshold=0.7)

# Ask questions
answer = chatbot.query("What's your return policy?")
print(answer)
```

### Benchmark Testing
```python
from benchmark_testing import BenchmarkTester

# Initialize tester
tester = BenchmarkTester('your_faq.csv', evaluation_mode='cosine')

# Run benchmark on 10 random questions
results = tester.run_benchmark(10)

# Print results
tester.print_results(results)
```

## 🔍 Troubleshooting

### Common Issues

1. **"GROQ_API_KEY environment variable not found"**
   - Solution: Set your Groq API key as an environment variable

2. **"CSV file must have at least 2 columns"**
   - Solution: Ensure your CSV has question and answer columns

3. **"I couldn't find a relevant answer"**
   - Solution: Lower the similarity threshold or rephrase the question

4. **Slow response times**
   - Solution: Reduce the size of your FAQ dataset or use a faster embedding model

### Performance Tips

- **Optimal FAQ Size**: 50-500 question-answer pairs for best performance
- **Question Quality**: Clear, specific questions work better than vague ones
- **Answer Length**: Concise answers (50-200 words) are most effective
- **Similarity Threshold**: Start with 0.7 and adjust based on your needs

## 🤝 Contributing

We welcome contributions to improve the FAQ chatbot! Here's how you can help:

### Areas for Contribution
- **New Features**: Additional evaluation metrics, UI improvements, advanced RAG techniques
- **Bug Fixes**: Performance issues, edge cases, error handling
- **Documentation**: Examples, tutorials, API documentation
- **Testing**: Additional test cases, integration tests

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes with proper documentation
4. Add tests for new functionality
5. Submit a pull request with a clear description

### Code Style
- Follow PEP 8 Python style guidelines
- Add docstrings for all functions and classes
- Include type hints where appropriate
- Write clear, self-documenting code

## 📄 License

This project is open source and available under the MIT License.

## 🙋‍♂️ Support

For questions, issues, or feature requests:
1. Check the troubleshooting section above
2. Search existing issues in the repository
3. Create a new issue with detailed description
4. Include error messages and system information

## 🔮 Future Enhancements

- **Multi-language Support**: Support for non-English FAQs
- **Advanced RAG Techniques**: Hybrid search, re-ranking, query expansion
- **Database Integration**: PostgreSQL, MongoDB support for larger datasets
- **Analytics Dashboard**: Usage statistics, popular questions, performance metrics
- **API Endpoint**: REST API for integration with other systems
- **Voice Interface**: Speech-to-text and text-to-speech capabilities

## Prompt Used for Generating Initial files**
```
You are an expert in building Generative AI apps using Retrieval Augmented Generation (RAG) approach. Please help build such RAG based chatbot app in python programming to answer questions from FAQs stored in a CSV file having two columns one for questions and the other for the corresponding anawes. Please generate following files:

main_faq_chatbot.py:  main chatbot class which has rag approach on faq CSV. Uses llamaindex library and groq llm API. It vectorizes and indexes questions from the first column and stories corresponding answers each vector along with the metadata. When a query comes, it is vectorized and matched to vectors of questions only, answe for the most similar queston, above threshold is given as the answer. The _main_ below the file tests a couple of questions similar to questions in the csv. 

streamlit _main-py: it has basic streamlit chatbot ui with facility to upload CSV on the left panel and chatbot interaction boxes in the main panel, this should call just main query function from faq class apart from the constructor.

benchmark_testing.py: it runs the query function from the FAQ class on random 10 questions in the csv, gets the responses and compares with the actual answers from the csv by two ways..one by using embedding plus cosine similarity and the other with llm prompt as a judge. default mode is cosine similarity. Use pandas library to read the csv and select random 10 questions-answers pairs for bench-marking.

README.md: standard GitHub readme with introduction, explanation of the files, how to use and contribution sections.

Please review all these files a few times for consistency before outputting. Keep code simple and well documented.

```
