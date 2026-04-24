# https://docs.llamaindex.ai/en/stable/use_cases/prompting/

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.prompts import RichPromptTemplate

import nest_asyncio

nest_asyncio.apply()

llm = Groq(model="llama3-8b-8192")
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
Settings.llm = llm
Settings.embed_model = embed_model

template_str = """We have provided context information below.
---------------------
{{ context_str }}
---------------------
Given this information, please answer the question: {{ query_str }}
"""

qa_template = RichPromptTemplate(template_str)

# you can create text prompt (for completion API)
prompt = qa_template.format(context_str="You are an expert in AI", query_str="What are components of AI?")

response = llm.complete(prompt)
print(response.text)

# # or easily convert to message prompts (for chat API)
# messages = qa_template.format_messages(context_str="You are an expert in AI", query_str="What are components of AI?")
# response = llm.chat(messages)
# print(response.text)