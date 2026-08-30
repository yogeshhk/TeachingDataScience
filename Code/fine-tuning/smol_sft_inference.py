import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline


# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps" if torch.backends.mps.is_available() else "cpu"
# )
#

def load_finetuned_model(model_path):
    """
    Load a fine-tuned model and its tokenizer for inference

    Args:
        model_path (str): Path to the saved model directory
        device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
        pipeline: A transformers pipeline ready for inference
    """
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16  # Use float16 for efficiency
    )

    # Create pipeline for text generation
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    return generator


def generate_text(generator, prompt, max_length=100, num_return_sequences=1):
    """
    Generate text using the fine-tuned model

    Args:
        generator: The pipeline object
        prompt (str): Input text to generate from
        max_length (int): Maximum length of generated text
        num_return_sequences (int): Number of different sequences to generate

    Returns:
        list: Generated text sequences
    """
    outputs = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        pad_token_id=generator.tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

    return [output['generated_text'] for output in outputs]


# Example usage
if __name__ == "__main__":
    finetune_name = "SmolLM2-FT-MyDataset"
    # Path to your saved model
    model_path = f"./models/{finetune_name}"

    # Load the model
    generator = load_finetuned_model(model_path)

    # Example prompt
    prompt = "Write a haiku about programming"

    # Generate text
    generated_texts = generate_text(
        generator,
        prompt,
        max_length=100,
        num_return_sequences=1
    )

    # Print results
    for i, text in enumerate(generated_texts):
        print(f"Generated text {i + 1}:")
        print(text)
        print("-" * 50)
