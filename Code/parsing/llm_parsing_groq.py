import os
from groq import Groq
import json

class GroqResumeParser:
    """
    A class to parse resumes using the GROQ API with Google's Gemma LLM.
    """
    def __init__(self, api_key: str, model: str = "gemma-7b-it"):
        """
        Initializes the GroqResumeParser.

        Args:
            api_key (str): Your Groq API key.
            model (str): The model to use for parsing.
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.client = Groq(api_key=api_key)
        self.model = model

    def parse_resume_text(self, resume_text: str) -> dict:
        """
        Parses resume text to extract structured information.

        Args:
            resume_text (str): The raw text of the resume.

        Returns:
            dict: A dictionary containing the parsed information.
        """
        prompt = f"""
        You are an expert resume parser. Extract the following information from the resume text provided below:
        - Name
        - Contact Information (Email, Phone)
        - Summary
        - Work Experience (Job Title, Company, Dates, Responsibilities)
        - Education (Degree, University, Graduation Year)
        - Skills

        Return the output in a clean JSON format.

        Resume Text:
        ---
        {resume_text}
        ---
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
            )
            response_text = chat_completion.choices[0].message.content
            return json.loads(response_text)
        except Exception as e:
            print(f"An error occurred during API call: {e}")
            return None

if __name__ == '__main__':
    # This is a conceptual test case.
    # You need a valid Groq API key to run this.
    
    # Replace 'YOUR_GROQ_API_KEY' with your actual API key
    # It's recommended to use environment variables for API keys
    api_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY")

    if api_key == "YOUR_GROQ_API_KEY":
        print("Please set the GROQ_API_KEY environment variable or replace 'YOUR_GROQ_API_KEY' in the script.")
    else:
        parser = GroqResumeParser(api_key=api_key)
        
        sample_resume = """
        Jane Smith
        Data Scientist
        jane.smith@email.com | (123) 456-7890

        Summary:
        Highly skilled Data Scientist with 5 years of experience in machine learning and data analysis.

        Experience:
        - Senior Data Scientist, TechCorp (2020-Present)
          - Led projects on predictive modeling.
        - Data Analyst, Data Inc. (2018-2020)
          - Analyzed large datasets to extract insights.

        Education:
        - M.S. in Computer Science, University of Tech (2018)
        
        Skills: Python, R, Machine Learning, SQL
        """
        
        print("--- Parsing sample resume with Groq ---")
        parsed_data = parser.parse_resume_text(sample_resume)
        
        if parsed_data:
            print(json.dumps(parsed_data, indent=2))
        else:
            print("Failed to parse the resume.")