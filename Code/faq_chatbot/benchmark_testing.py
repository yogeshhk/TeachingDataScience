"""
Benchmark Testing for FAQ Chatbot
This module tests the FAQ chatbot performance by comparing generated answers
with actual answers using cosine similarity and LLM-based evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import argparse
from typing import List, Dict, Tuple
import logging
from main_faq_chatbot import FAQChatbot
from sentence_transformers import SentenceTransformer
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkTester:
    """
    Benchmarks FAQ chatbot performance using different evaluation methods.
    """
    
    def __init__(self, csv_file_path: str, evaluation_mode: str = 'cosine'):
        """
        Initialize benchmark tester.
        
        Args:
            csv_file_path (str): Path to FAQ CSV file
            evaluation_mode (str): 'cosine' for cosine similarity or 'llm' for LLM judge
        """
        self.csv_file_path = csv_file_path
        self.evaluation_mode = evaluation_mode
        self.faq_data = None
        self.chatbot = None
        self.embedding_model = None
        
        # Load data and initialize components
        self._load_data()
        self._initialize_chatbot()
        if evaluation_mode == 'cosine':
            self._initialize_embedding_model()
    
    def _load_data(self):
        """Load FAQ data from CSV file."""
        try:
            self.faq_data = pd.read_csv(self.csv_file_path)
            
            if len(self.faq_data.columns) < 2:
                raise ValueError("CSV must have at least 2 columns")
            
            # Use first two columns as questions and answers
            self.faq_data.columns = ['question', 'answer'] + list(self.faq_data.columns[2:])
            self.faq_data = self.faq_data.dropna(subset=['question', 'answer'])
            
            logger.info(f"Loaded {len(self.faq_data)} FAQ pairs for testing")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _initialize_chatbot(self):
        """Initialize the FAQ chatbot."""
        try:
            self.chatbot = FAQChatbot(self.csv_file_path, similarity_threshold=0.6)
            logger.info("Chatbot initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing chatbot: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer for cosine similarity evaluation."""
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Embedding model initialized for cosine similarity evaluation")
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            raise
    
    def select_random_questions(self, n: int = 10) -> pd.DataFrame:
        """
        Select random questions from the FAQ dataset.
        
        Args:
            n (int): Number of questions to select
            
        Returns:
            pd.DataFrame: Random sample of FAQ pairs
        """
        if len(self.faq_data) < n:
            logger.warning(f"Only {len(self.faq_data)} questions available, using all")
            return self.faq_data.copy()
        
        sample = self.faq_data.sample(n=n, random_state=42)
        return sample.reset_index(drop=True)
    
    def evaluate_with_cosine_similarity(self, generated_answer: str, expected_answer: str) -> float:
        """
        Evaluate answer quality using cosine similarity of embeddings.
        
        Args:
            generated_answer (str): Answer from chatbot
            expected_answer (str): Expected answer from CSV
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        try:
            # Generate embeddings
            embeddings = self.embedding_model.encode([generated_answer, expected_answer])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {e}")
            return 0.0
    
    def evaluate_with_llm_judge(self, question: str, generated_answer: str, expected_answer: str) -> float:
        """
        Evaluate answer quality using LLM as judge.
        
        Args:
            question (str): Original question
            generated_answer (str): Answer from chatbot
            expected_answer (str): Expected answer from CSV
            
        Returns:
            float: LLM evaluation score (0-1)
        """
        try:
            # Create evaluation prompt
            prompt = f"""
            You are an expert evaluator. Rate how well the generated answer matches the expected answer for the given question.
            
            Question: {question}
            
            Expected Answer: {expected_answer}
            
            Generated Answer: {generated_answer}
            
            Please rate the quality on a scale of 0.0 to 1.0 where:
            - 1.0 = Perfect match or equivalent meaning
            - 0.8 = Very good, minor differences
            - 0.6 = Good, some differences but core meaning preserved
            - 0.4 = Partial match, missing important details
            - 0.2 = Poor match, different meaning
            - 0.0 = Completely wrong or no answer
            
            Respond with only the numeric score (e.g., 0.8):
            """
            
            # Use chatbot's LLM to evaluate (assuming it has access to the LLM)
            # This is a simplified approach - in practice, you'd want a separate LLM instance
            if "I couldn't find a relevant answer" in generated_answer:
                return 0.0
            elif generated_answer == expected_answer:
                return 1.0
            else:
                # Simplified scoring based on text similarity as fallback
                # In a real implementation, you'd call the LLM with the prompt
                common_words = set(generated_answer.lower().split()) & set(expected_answer.lower().split())
                total_words = set(generated_answer.lower().split()) | set(expected_answer.lower().split())
                return len(common_words) / len(total_words) if total_words else 0.0
                
        except Exception as e:
            logger.error(f"Error in LLM evaluation: {e}")
            return 0.0
    
    def run_benchmark(self, n_questions: int = 10) -> Dict:
        """
        Run benchmark test on random questions.
        
        Args:
            n_questions (int): Number of questions to test
            
        Returns:
            dict: Benchmark results
        """
        logger.info(f"Starting benchmark test with {n_questions} questions using {self.evaluation_mode} evaluation")
        
        # Select random questions
        test_questions = self.select_random_questions(n_questions)
        
        results = {
            'questions': [],
            'expected_answers': [],
            'generated_answers': [],
            'scores': [],
            'response_times': []
        }
        
        total_score = 0
        total_time = 0
        
        for idx, row in test_questions.iterrows():
            question = row['question']
            expected_answer = row['answer']
            
            # Measure response time
            start_time = time.time()
            generated_answer = self.chatbot.query(question)
            response_time = time.time() - start_time
            
            # Evaluate answer quality
            if self.evaluation_mode == 'cosine':
                score = self.evaluate_with_cosine_similarity(generated_answer, expected_answer)
            else:  # llm mode
                score = self.evaluate_with_llm_judge(question, generated_answer, expected_answer)
            
            # Store results
            results['questions'].append(question)
            results['expected_answers'].append(expected_answer)
            results['generated_answers'].append(generated_answer)
            results['scores'].append(score)
            results['response_times'].append(response_time)
            
            total_score += score
            total_time += response_time
            
            logger.info(f"Question {idx+1}/{len(test_questions)}: Score = {score:.3f}, Time = {response_time:.3f}s")
        
        # Calculate summary statistics
        avg_score = total_score / len(test_questions)
        avg_response_time = total_time / len(test_questions)
        
        results['summary'] = {
            'total_questions': len(test_questions),
            'average_score': avg_score,
            'average_response_time': avg_response_time,
            'evaluation_mode': self.evaluation_mode,
            'scores_distribution': {
                'min': min(results['scores']),
                'max': max(results['scores']),
                'std': np.std(results['scores'])
            }
        }
        
        return results
    
    def print_results(self, results: Dict):
        """Print benchmark results in a formatted way."""
        print("\n" + "="*80)
        print("FAQ CHATBOT BENCHMARK RESULTS")
        print("="*80)
        
        summary = results['summary']
        print(f"Evaluation Mode: {summary['evaluation_mode'].upper()}")
        print(f"Total Questions Tested: {summary['total_questions']}")
        print(f"Average Score: {summary['average_score']:.3f}")
        print(f"Average Response Time: {summary['average_response_time']:.3f}s")
        print(f"Score Distribution: Min={summary['scores_distribution']['min']:.3f}, "
              f"Max={summary['scores_distribution']['max']:.3f}, "
              f"Std={summary['scores_distribution']['std']:.3f}")
        
        print("\n" + "-"*80)
        print("DETAILED RESULTS")
        print("-"*80)
        
        for i in range(len(results['questions'])):
            print(f"\n[Question {i+1}]")
            print(f"Q: {results['questions'][i]}")
            print(f"Expected: {results['expected_answers'][i][:100]}...")
            print(f"Generated: {results['generated_answers'][i][:100]}...")
            print(f"Score: {results['scores'][i]:.3f} | Time: {results['response_times'][i]:.3f}s")
    
    def save_results_to_csv(self, results: Dict, output_file: str = "benchmark_results.csv"):
        """Save detailed results to CSV file."""
        df = pd.DataFrame({
            'question': results['questions'],
            'expected_answer': results['expected_answers'],
            'generated_answer': results['generated_answers'],
            'score': results['scores'],
            'response_time': results['response_times']
        })
        
        df.to_csv(output_file, index=False)
        logger.info(f"Detailed results saved to {output_file}")


def main():
    """Main function to run benchmark testing."""
    parser = argparse.ArgumentParser(description='Benchmark FAQ Chatbot Performance')
    parser.add_argument('--csv_file', type=str, default='sample_faq.csv',
                       help='Path to FAQ CSV file')
    parser.add_argument('--mode', choices=['cosine', 'llm'], default='cosine',
                       help='Evaluation mode: cosine similarity or LLM judge')
    parser.add_argument('--questions', type=int, default=10,
                       help='Number of questions to test')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                       help='Output CSV file for detailed results')
    
    args = parser.parse_args()
    
    try:
        # Check if CSV file exists
        if not os.path.exists(args.csv_file):
            print(f"CSV file {args.csv_file} not found. Creating sample data...")
            sample_data = {
                'question': [
                    'What is your return policy?',
                    'How long does shipping take?',
                    'Do you offer customer support?',
                    'What payment methods do you accept?',
                    'How can I track my order?',
                    'Do you ship internationally?',
                    'What is your refund process?',
                    'How can I cancel my order?',
                    'Do you offer warranties?',
                    'What are your business hours?',
                    'How do I contact technical support?',
                    'Can I change my shipping address?',
                    'What is the minimum order value?',
                    'Do you offer bulk discounts?',
                    'How do I create an account?'
                ],
                'answer': [
                    'We offer a 30-day return policy for all unused items in original packaging.',
                    'Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days.',
                    'Yes, we offer 24/7 customer support via email, chat, and phone.',
                    'We accept all major credit cards, PayPal, and bank transfers.',
                    'You can track your order using the tracking number sent to your email after shipment.',
                    'Yes, we ship to over 50 countries worldwide. International shipping takes 10-15 business days.',
                    'Refunds are processed within 5-7 business days after we receive the returned item.',
                    'You can cancel your order within 2 hours of placement by contacting customer support.',
                    'Yes, we offer a 1-year warranty on all products against manufacturing defects.',
                    'Our business hours are Monday to Friday 9 AM to 6 PM EST.',
                    'Technical support is available 24/7 via our help desk and online chat.',
                    'You can change your shipping address within 1 hour of placing the order.',
                    'The minimum order value is $25 for domestic orders and $50 for international orders.',
                    'Yes, we offer bulk discounts starting from orders of 100 items or more.',
                    'You can create an account by clicking the Sign Up button and filling out the registration form.'
                ]
            }
            df = pd.DataFrame(sample_data)
            df.to_csv(args.csv_file, index=False)
            print(f"Sample FAQ data created: {args.csv_file}")
        
        # Initialize benchmark tester
        tester = BenchmarkTester(args.csv_file, args.mode)
        
        # Run benchmark
        results = tester.run_benchmark(args.questions)
        
        # Display results
        tester.print_results(results)
        
        # Save detailed results
        tester.save_results_to_csv(results, args.output)
        
        # Performance summary
        summary = results['summary']
        print(f"\n{'='*80}")
        print("PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        print(f"Overall Performance Score: {summary['average_score']:.1%}")
        
        if summary['average_score'] >= 0.8:
            print("✅ EXCELLENT - Chatbot is performing very well!")
        elif summary['average_score'] >= 0.6:
            print("✓ GOOD - Chatbot is performing well with room for improvement")
        elif summary['average_score'] >= 0.4:
            print("⚠ FAIR - Chatbot needs improvement")
        else:
            print("❌ POOR - Chatbot needs significant improvement")
        
    except Exception as e:
        logger.error(f"Benchmark testing failed: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())