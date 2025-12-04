from langchain_core.prompts import PromptTemplate

# Use in chain with Groq
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

template = """You are a {role} assistant.
Task: {task}
Context: {context}
Provide a {format} response."""

prompt = PromptTemplate(
    template=template,
    input_variables=["role", "task", "context", "format"]
)

formatted = prompt.format(
    role="helpful",
    task="explain quantum computing",
    context="for beginners",
    format="simple"
)

chain = prompt | ChatGroq(model="llama-3.3-70b-versatile") | StrOutputParser()
result = chain.invoke({
    "role": "helpful",
    "task": "explain quantum computing",
    "context": "for beginners",
    "format": "simple"
})

print(result)

# Chat Prompt Template:
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}"),
    ("human", "{input}"),
    ("ai", "I understand. Let me help with that."),
    ("human", "{follow_up}")
])

chain = prompt | ChatGroq(model="llama-3.3-70b-versatile") | StrOutputParser()
result = chain.invoke({
    "role": "helpful",
    "input": "Explain quantum computing",
    "follow_up": "Make it simpler"
})
print(result)