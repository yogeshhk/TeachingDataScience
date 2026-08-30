# https://medium.com/@rajveer.rathod1301/getting-started-with-sarvam-ai-a-complete-python-sdk-guide-37dd33fbb60b

import os
from sarvamai import SarvamAI

def initialize_client():
    api_key = os.getenv("SARVAM_API_KEY")
    client = SarvamAI(api_subscription_key=api_key)
    return client
    
def generate_text(client, prompt, max_tokens=100):
    try:
        response = client.chat.completions(
            model="sarvam-m",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in text generation: {e}")
        return None
        
def translate_text(text, source_lang="en-IN", target_lang="hi-IN"):
    try:
        client = SarvamAI(api_subscription_key=api_key)
        response = client.translate(
            model="sarvam-m",
            input=text,
            source_language=source_lang,
            target_language=target_lang
        )
        return response.translated_text
    except Exception as e:
        print(f"Error in translation: {e}")
        return None

def summarize_text(client, text):
    try:
        prompt = f"Please summarize the following text:\n\n{text}"
        response = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            model="sarvam-m",
            max_tokens=150,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in summarization: {e}")
        return None        
        
def ask_question(client, context, question):
    try:
        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        response = client.chat.completions(
            messages=[{"role": "user", "content": prompt}],
            model="sarvam-m",
            max_tokens=100,
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in question answering: {e}")
        return None        
        
def main():
    client = initialize_client()
    print("=== Sarvam AI SDK Demo ===\n")
    
    # Text Generation
    poem = generate_text(client, "Write a short poem about artificial intelligence")
    print(f"Poem : {poem}")
    
    # Summarization
    long_text = "Artificial Intelligence (AI) is a branch of computer science..."
    summary = summarize_text(client, long_text)
    print(f"Summary : {summary}")
    
    # Question Answering
    ans = ask_question(client,
                       "Python is a high-level programming language known for its simplicity and readability.",
                       "What is Python known for?")
    print(f"QnA Answer: {ans}")

if __name__ == "__main__":
    main()                       