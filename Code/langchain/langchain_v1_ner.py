from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Setup Model
model = ChatGroq(model="llama-3.3-70b-versatile")

# 2. Create Prompt for NER extraction
template = """
Identify the named entities (Person, Organization, Amount) in the text.
Format the output as a bulleted list.

Text: {text}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. Build Chain
chain = prompt | model | StrOutputParser()

# 4. Invoke
input_text = "Microsoft is acquiring Nuance Communications for $19.7 billion."
entities = chain.invoke({"text": input_text})
print(entities)