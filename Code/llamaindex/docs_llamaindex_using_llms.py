# https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.llms import ChatMessage,  TextBlock, ImageBlock
from llama_index.core import VectorStoreIndex

import nest_asyncio

nest_asyncio.apply()

llm = Groq(model="llama3-8b-8192")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

# response = llm.complete("William Shakespeare is ")
# print(response)

# handle = llm.stream_complete("William Shakespeare is ")

# for token in handle:
#     print(token.delta, end="", flush=True)

# messages = [
#     ChatMessage(role="system", content="You are a helpful assistant."),
#     ChatMessage(role="user", content="Tell me a joke."),
# ]
# chat_response = llm.chat(messages)
# print(chat_response)

# messages = [
#     ChatMessage(
#         role="user",
#         blocks=[
#             ImageBlock(path="image.png"),
#             TextBlock(text="Describe the image in a few sentences."),
#         ],
#     )
# ]

# resp = llm.chat(messages)
# print(resp.message.content)

# Some LLMs (OpenAI, Anthropic, Gemini, Ollama, etc.) support tool calling directly over API calls
from llama_index.core.tools import FunctionTool


def generate_song(name: str, artist: str) -> str:
    """Generates a song with provided name and artist."""
    return {"name": name, "artist": artist}


tool = FunctionTool.from_defaults(fn=generate_song)

response = llm.predict_and_call(
    [tool],
    "Pick a random song for me",
)
print(str(response))