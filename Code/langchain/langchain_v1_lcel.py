from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Set up Groq model, e.g., Gemma or Llama 3
llm = ChatGroq(model_name="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
output_parser = StrOutputParser()

# Build chain with pipe operator
chain = prompt | llm | output_parser

# # Invoke the chain
# result = chain.invoke({"topic": "LangChain"})
# print(result)

# # Invoke the Streaming Support
import asyncio

async def main():
    print("--- 1. Synchronous Streaming ---")
    # Stream tokens as they're generated (Works in standard functions)
    for chunk in chain.stream({"topic": "AI"}):
        print(chunk, end="", flush=True)
    print("\n")

    print("--- 2. Async Invocation ---")
    # Async invocation (Must be inside async function)
    result = await chain.ainvoke({"topic": "AI"})
    print(f"Result: {result}\n")

    print("--- 3. Async Streaming ---")
    # Async streaming
    async for chunk in chain.astream({"topic": "AI"}):
        print(chunk, end="", flush=True)
    print("\n")

    print("--- 4. Batch Processing ---")
    # Process multiple inputs in parallel (Sync version)
    results = chain.batch([
        {"topic": "AI"},
        {"topic": "ML"},
        {"topic": "LangChain"}
    ])
    print(f"Batch Results: {results}")

# Run the async event loop
if __name__ == "__main__":
    asyncio.run(main())
