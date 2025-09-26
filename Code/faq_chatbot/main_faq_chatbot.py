"""
FAQ Chatbot using RAG approach with LlamaIndex and Groq LLM
This module implements a Retrieval Augmented Generation (RAG) based chatbot
for answering questions from FAQ data stored in CSV format.
"""

import os
import pandas as pd
from typing import List, Optional, Tuple
import logging
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.groq import Groq
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FAQChatbot:
    """
    A RAG-based FAQ chatbot that uses vector similarity to match user queries
    with FAQ questions and returns corresponding answers.
    """
    
    def __init__(self, csv_file_path: str, similarity_threshold: float = 0.7):
        """
        Initialize the FAQ chatbot with CSV data.
        
        Args:
            csv_file_path (str): Path to the CSV file containing FAQ data
            similarity_threshold (float): Minimum similarity score for answer retrieval
        """
        self.csv_file_path = csv_file_path
        self.similarity_threshold = similarity_threshold
        self.faq_data = None
        self.index = None
        self.query_engine = None
        
        # Configure LlamaIndex settings
        self._setup_llm_and_embeddings()
        
        # Load and process FAQ data
        self._load_faq_data()
        self._create_vector_index()
        
    # def _setup_llm_and_embeddings(self):
    #     """Configure LLM and embedding models for LlamaIndex."""
    #     try:
    #         # Initialize Groq LLM
    #         groq_api_key = os.getenv('GROQ_API_KEY')
    #         if not groq_api_key:
    #             raise ValueError("GROQ_API_KEY environment variable not found")
            
    #         llm = Groq(model="mixtral-8x7b-32768", api_key=groq_api_key)
            
    #         # Use HuggingFace embedding model (free alternative)
    #         embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
    #         # Set global settings
    #         Settings.llm = llm
    #         Settings.embed_model = embed_model
            
    #         logger.info("LLM and embedding models configured successfully")
            
    #     except Exception as e:
    #         logger.error(f"Error setting up LLM and embeddings: {e}")
    #         raise

    def _setup_llm_and_embeddings(self):
        """Configure LLM and embedding models for LlamaIndex."""
        try:
            # Initialize Hugging Face LLM with Gemma
            hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not hf_api_key:
                raise ValueError("HUGGINGFACE_API_KEY environment variable not found")
            
            llm = HuggingFaceLLM(
                model_name="google/gemma-2b-it",
                tokenizer_name="google/gemma-2b-it",
                token=hf_api_key,
                context_window=2048,
                max_new_tokens=512,
                generate_kwargs={"temperature": 0.1, "do_sample": True}
            )
            
            # Use HuggingFace embedding model (free alternative)
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            # Set global settings
            Settings.llm = llm
            Settings.embed_model = embed_model
            
            logger.info("LLM and embedding models configured successfully")
            
        except Exception as e:
            logger.error(f"Error setting up LLM and embeddings: {e}")
            raise
        
    def _load_faq_data(self):
        """Load FAQ data from CSV file."""
        try:
            self.faq_data = pd.read_csv(self.csv_file_path)
            
            # Validate CSV structure
            if len(self.faq_data.columns) < 2:
                raise ValueError("CSV file must have at least 2 columns (Question and Answer)")
            
            # Use first two columns as questions and answers
            self.faq_data.columns = ['Question', 'Answer'] + list(self.faq_data.columns[2:])
            
            # Remove rows with empty questions or answers
            self.faq_data = self.faq_data.dropna(subset=['Question', 'Answer'])
            
            logger.info(f"Loaded {len(self.faq_data)} FAQ pairs from {self.csv_file_path}")
            
        except Exception as e:
            logger.error(f"Error loading FAQ data: {e}")
            raise
    
    def _create_vector_index(self):
        """Create vector index from FAQ questions with answers as metadata."""
        try:
            documents = []
            
            for idx, row in self.faq_data.iterrows():
                # Create document with question as content and answer as metadata
                doc = Document(
                    text=row['question'],
                    metadata={
                        'answer': row['answer'],
                        'question_id': idx,
                        'original_question': row['question']
                    }
                )
                documents.append(doc)
            
            # Create vector index
            self.index = VectorStoreIndex.from_documents(documents)
            
            # Create query engine with similarity threshold
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=3
            )
            
            self.query_engine = RetrieverQueryEngine(
                retriever=retriever,
                postprocessors=[SimilarityPostprocessor(similarity_cutoff=self.similarity_threshold)]
            )
            
            logger.info("Vector index created successfully")
            
        except Exception as e:
            logger.error(f"Error creating vector index: {e}")
            raise
    
    def query(self, user_question: str) -> str:
        """
        Process user query and return the most relevant answer.
        
        Args:
            user_question (str): User's question
            
        Returns:
            str: Answer from FAQ or fallback message
        """
        try:
            if not user_question.strip():
                return "Please provide a valid question."
            
            # Query the vector index
            retriever = VectorIndexRetriever(
                index=self.index,
                similarity_top_k=1
            )
            
            nodes = retriever.retrieve(user_question)
            
            if nodes and len(nodes) > 0:
                # Get the most similar node
                best_node = nodes[0]
                
                # Check similarity threshold (score is typically between 0 and 1)
                if best_node.score >= self.similarity_threshold:
                    answer = best_node.node.metadata.get('answer', 'No answer found.')
                    return answer
                else:
                    return f"I couldn't find a relevant answer to your question. Please try rephrasing or contact support for assistance."
            else:
                return "I couldn't find a relevant answer to your question. Please try rephrasing or contact support for assistance."
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return "Sorry, I encountered an error while processing your question. Please try again."
    
    def get_faq_stats(self) -> dict:
        """
        Get statistics about the loaded FAQ data.
        
        Returns:
            dict: Statistics about FAQ data
        """
        if self.faq_data is not None:
            return {
                'total_faqs': len(self.faq_data),
                'avg_question_length': self.faq_data['question'].str.len().mean(),
                'avg_answer_length': self.faq_data['answer'].str.len().mean(),
                'similarity_threshold': self.similarity_threshold
            }
        return {}


def main():
    """Test the FAQ chatbot with sample questions."""

    faqs_file_path = "data/BankFAQs.csv"
    
    try:
        # Initialize chatbot
        print("Initializing FAQ Chatbot...")
        chatbot = FAQChatbot(faqs_file_path, similarity_threshold=0.6)
        
        # Display FAQ stats
        stats = chatbot.get_faq_stats()
        print(f"\nFAQ Statistics: {stats}")
        
        # Test queries
        test_questions = [
            "What is the validity of the OTP?",             # Similar to existing question
            "What is a Personal Greeting?",                 # Similar to greetings question
            "How do I repay my Business Loan?",             # Similar to customer support question
            "What is RTGS?",                                # Similar to NEFT question
            "Do I need a co-applicant for a Housing Loan?"  # Unrelated question
        ]
        
        print("\n" + "="*50)
        print("TESTING FAQ CHATBOT")
        print("="*50)
        
        for question in test_questions:
            print(f"\nQ: {question}")
            answer = chatbot.query(question)
            print(f"A: {answer}")
            print("-" * 50)
    
    except Exception as e:
        print(f"Error: {e}")
        print("Please make sure you have set the HUGGINGFACE_API_KEY environment variable")


if __name__ == "__main__":
    main()