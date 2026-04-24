"""
LangGraph agent configuration and setup.
"""
from typing import List
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver


# System prompt for the document intelligence assistant
SYSTEM_PROMPT = """You are a helpful document intelligence assistant. You have access to documents that have been uploaded and processed (PDFs, Word documents, presentations, HTML files, etc.).

GUIDELINES:
- Use the search_documents tool to find relevant information
- Be efficient: one well-crafted search is usually sufficient
- Only search again if the first results are clearly incomplete
- Provide clear, accurate answers based on the document contents
- Always cite your sources with filenames or document titles
- If information isn't found, say so clearly
- Be concise but thorough

When answering:
1. Search the documents with a focused query
2. Synthesize a clear answer from the results
3. Include source citations (filenames)
4. Only search again if absolutely necessary
"""


def create_documentation_agent(tools: List[BaseTool], model_name: str = "gpt-4o-mini"):
    """
    Create a document intelligence assistant agent using LangGraph.

    Args:
        tools: List of tools the agent can use
        model_name: Name of the OpenAI model to use

    Returns:
        A configured LangGraph agent
    """
    # Initialize the language model
    llm = ChatOpenAI(model=model_name, temperature=0)

    # Create a memory saver for conversation history
    memory = MemorySaver()

    # Create the ReAct agent
    agent = create_react_agent(
        llm,
        tools=tools,
        prompt=SYSTEM_PROMPT,
        checkpointer=memory
    )

    return agent
