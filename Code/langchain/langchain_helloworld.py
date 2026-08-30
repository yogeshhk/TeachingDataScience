from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.7)
template = """You are a teacher in physics for High School student. Given the text of question, it is your job to write a answer that question with example.
Question: {text}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
answer_chain = prompt | llm | StrOutputParser()
answer = answer_chain.invoke({"text": "What is the formula for Gravitational Potential Energy (GPE)?"})
print(answer)
