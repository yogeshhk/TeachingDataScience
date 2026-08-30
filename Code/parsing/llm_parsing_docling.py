import os
from docling import Docling
from docling.datamodel.document_and_entity_response import DocumentAndEntityResponse

class DoclingResumeParser:
    """
    A class to parse resumes using the Docling API.
    """
    def __init__(self, api_key: str):
        """
        Initializes the DoclingResumeParser.

        Args:
            api_key (str): Your Docling API key.
        """
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.client = Docling(api_key=api_key)

    def parse_resume(self, file_path: str) -> DocumentAndEntityResponse:
        """
        Parses a single resume file.

        Args:
            file_path (str): The path to the resume file.

        Returns:
            DocumentAndEntityResponse: The parsed resume data.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")
        
        try:
            with open(file_path, "rb") as f:
                response = self.client.document_and_entity.upload(file=f)
            return response
        except Exception as e:
            print(f"An error occurred while parsing {file_path}: {e}")
            return None

if __name__ == '__main__':
    # This is a conceptual test case.
    # To run this, you need a 'data' folder with resume files
    # and a valid Docling API key.
    
    # Create a dummy data directory and a dummy file for demonstration
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/sample_resume.txt', 'w') as f:
        f.write("John Doe\nSoftware Engineer\nSan Francisco, CA\n\nSkills: Python, Java, SQL")

    # Replace 'YOUR_DOCLING_API_KEY' with your actual API key
    # For security, it's better to load this from an environment variable
    api_key = os.environ.get("DOCLING_API_KEY", "YOUR_DOCLING_API_KEY")

    if api_key == "YOUR_DOCLING_API_KEY":
        print("Please set the DOCLING_API_KEY environment variable or replace 'YOUR_DOCLING_API_KEY' in the script.")
    else:
        parser = DoclingResumeParser(api_key=api_key)
        
        test_file = 'data/sample_resume.txt'
        
        print(f"--- Parsing {test_file} with Docling ---")
        parsed_data = parser.parse_resume(test_file)

        if parsed_data:
            print("Successfully parsed the resume.")
            # You can inspect the 'parsed_data' object for extracted entities
            # For example, to see extracted entities:
            if parsed_data.entities:
                for entity in parsed_data.entities:
                    print(f"- Type: {entity.type}, Value: {entity.value}")
            else:
                print("No entities were extracted.")
        else:
            print("Failed to parse the resume.")