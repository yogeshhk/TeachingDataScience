from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.prompts.system import SHAKESPEARE_WRITING_ASSISTANT
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
from dotenv import load_dotenv
import os

load_dotenv()

llm = OpenAI(model="gpt-4o", temperature=0)
data = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data)


chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
# response = chat_engine.chat("What did Paul Graham do after YC?")

while True:
    text_input = input("User: ")
    if text_input == "exit":
        break
    response = chat_engine.chat(text_input)
    print(f"Agent: {response}")


# for token in response.response_gen:
#     print(token, end="")

# print(response)