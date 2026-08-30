import torch
print(torch.cuda.is_available())  # Should return True
print(torch.__version__)          # e.g., 2.3.0+cu121

# --- Import necessary LlamaIndex components ---
from llama_index.core import Settings
# from llama_index.llms.openai import OpenAI # For the LM Studio LLM endpoint
from llama_index.llms.lmstudio import LMStudio
# >>> Using HuggingFace embeddings locally is generally recommended with LM Studio <<<
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import nest_asyncio

nest_asyncio.apply()

# --- Configuration for LM Studio ---
# !! IMPORTANT: Verify these values with your LM Studio setup !!

# 1. API Base URL (Check LM Studio Server tab)
LM_STUDIO_API_BASE = "http://localhost:1234/v1" # Default for LM Studio

# 2. Model Identifier (CRITICAL: Get this from LM Studio after loading the model)
#    This is NOT just "llama3". Check the LM Studio server logs or UI for the exact ID.
#    Replace the placeholder below with the actual identifier.
LM_STUDIO_MODEL_NAME = "gemma-3-1b-it" # <<< REPLACE THIS PLACEHOLDER

# 3. API Key (LM Studio usually doesn't require one)
LM_STUDIO_API_KEY = "lm-studio" # Placeholder, often ignored by LM Studio

print(f"Configuring for LM Studio:")
print(f" - API Base: {LM_STUDIO_API_BASE}")
print(f" - Model Name: {LM_STUDIO_MODEL_NAME}")
print(f" - Using local HuggingFace embeddings.")

# --- Create the LLM instance pointing to LM Studio ---
llm = LMStudio(
    model_name=LM_STUDIO_MODEL_NAME, # Use the specific model identifier from LM Studio
    base_url=LM_STUDIO_API_BASE,
    # api_key=LM_STUDIO_API_KEY,
    # is_chat_model=True, # Llama 3 instruct models are chat models
    # LM Studio server might have timeout issues with complex tasks, increase if needed
    # timeout=600, # Example: 10 minutes, default is often 120 seconds
    temperature=0.7, # Adjust creativity/determinism
)

# --- Create a local Embedding model using HuggingFace ---
# LM Studio's focus is LLM serving, relying on its endpoint for embeddings might not work or be ideal.
# Using a separate local model is more robust.
# Requires: pip install llama-index-embeddings-huggingface sentence-transformers torch
print("Initializing local HuggingFace embedding model...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", # A good default, small and fast
    # device="cuda" # Uncomment this if you have a GPU and PyTorch with CUDA installed
    # device="cpu" # Explicitly use CPU if needed
)
print("Embedding model initialized.")

# --- Apply the configuration globally using Settings ---
Settings.llm = llm
Settings.embed_model = embed_model
print("LlamaIndex Settings configured.")
