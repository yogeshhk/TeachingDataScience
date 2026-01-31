import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.llms.groq import Groq
import faiss

# For demonstration, we'll use the Groq parser. You could also integrate the Docling parser.
from llm_parsing_groq import GroqResumeParser 

class ResumeRAG:
    """
    A class to build a RAG system for resumes.
    """
    def __init__(self, groq_api_key: str, data_dir: str = "data"):
        if not groq_api_key:
            raise ValueError("Groq API key is required.")
        
        self.data_dir = data_dir
        self.llm = Groq(model="llama3-8b-8192", api_key=groq_api_key)
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
        # Configure LlamaIndex settings
        Settings.llm = self.llm
        Settings.embed_model = self.embed_model
        
        self.index = self._build_index()

    def _build_index(self):
        """
        Builds the FAISS vector index from resumes in the data directory.
        """
        if not os.path.exists(self.data_dir):
            print(f"Data directory '{self.data_dir}' not found. Please create it and add resumes.")
            return None
            
        # Load documents
        reader = SimpleDirectoryReader(self.data_dir)
        documents = reader.load_data()

        if not documents:
            print("No documents found in the data directory.")
            return None

        # Semantic chunking
        splitter = SemanticSplitterNodeParser(
            buffer_size=1, breakpoint_percentile_threshold=95, embed_model=self.embed_model
        )
        nodes = splitter.get_nodes_from_documents(documents)

        # FAISS vector store setup
        d = 384  # Dimension of BAAI/bge-small-en-v1.5 embeddings
        faiss_index = faiss.IndexFlatL2(d)
        vector_store = FaissVectorStore(faiss_index=faiss_index)

        # Create the index
        index = VectorStoreIndex(nodes, vector_store=vector_store)
        return index

    def query(self, question: str):
        """
        Queries the RAG system.
        """
        if not self.index:
            return "The index has not been built. Please add resumes to the 'data' folder and restart."
            
        query_engine = self.index.as_query_engine()
        response = query_engine.query(question)
        return response

if __name__ == '__main__':
    # Create dummy data for testing
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/resume1.txt', 'w') as f:
        f.write("Alice Johnson is a project manager with 10 years of experience in agile methodologies.")
    with open('data/resume2.txt', 'w') as f:
        f.write("Bob Williams is a software developer skilled in Python and cloud computing on AWS.")

    # It's recommended to use environment variables for API keys
    api_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

    if api_key == "YOUR_GROQ_API_KEY":
        print("Please set GROQ_API_KEY environment variable or replace 'YOUR_GROQ_API_KEY'.")
    else:
        rag_system = ResumeRAG(groq_api_key=api_key)
        
        if rag_system.index:
            print("--- RAG System Test Cases ---")

            # Test case 1
            question1 = "Who has experience with agile?"
            print(f"Q: {question1}")
            answer1 = rag_system.query(question1)
            print(f"A: {answer1}\n")
            
            # Test case 2
            question2 = "List the skills of the software developer."
            print(f"Q: {question2}")
            answer2 = rag_system.query(question2)
            print(f"A: {answer2}\n")