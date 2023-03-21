from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain, PromptTemplate

template = """You are a teacher in physics for High School student. Given the text of question, it is your job to write a answer that question with example.
{chat_history}
Human: {question}
AI:
"""
prompt_template = PromptTemplate(input_variables=["chat_history","question"], template=template)
memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
    llm=OpenAI(),
    prompt=prompt_template,
    verbose=True,
    memory=memory,
)

llm_chain.predict(question="What is the formula for Gravitational Potential Energy (GPE)?")

result = llm_chain.predict(question="What is Joules?")
print(result)
