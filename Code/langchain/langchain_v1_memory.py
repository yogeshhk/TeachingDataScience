from langchain_classic.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Memory ---
memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="history"
)

# --- Prompt w/ memory placeholder ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# --- LLM ---
llm = ChatGroq(model="llama-3.3-70b-versatile")

# --- Runnable Chain with Memory Integration ---
chain = (
    RunnablePassthrough.assign(
        history=lambda x: memory.load_memory_variables({}).get("history", [])
    )
    | prompt
    | llm
    | StrOutputParser()
)

# Example conversation
user_input = "Hi, I'm Alice"
response = chain.invoke({"input": user_input})

# Save memory (required when using Runnable chains)
memory.save_context({"input": user_input}, {"output": response})

print(response)
