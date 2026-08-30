# Reference: https://levelup.gitconnected.com/langgraph-gemini-pro-custom-tool-streamlit-multi-agent-application-development-79c1473086b8
# https://www.youtube.com/watch?v=FWBnNcZv3kw
# https://github.com/langchain-ai/langgraph/blob/main/examples/multi_agent/multi-agent-collaboration.ipynb

# Installations
# pip install streamlit
# pip install langchainhub
# pip install langgraph
# pip install langchain_google_genai
# pip install -U langchain-openai langchain

# Assuming VERTEX AI set in Environment variables
from langchain import hub
from langchain.agents import Tool, create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import GoogleSerperAPIWrapper
import os
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
from typing import TypedDict, Annotated
from langchain_core.agents import AgentFinish
from langgraph.prebuilt.tool_executor import ToolExecutor
from langgraph.prebuilt import ToolInvocation
from langgraph.graph import END, StateGraph
from langchain_core.agents import AgentActionMessageLog
import streamlit as st

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_vertexai import VertexAI, ChatVertexAI

# from langchain.llms import VertexAI
# from langchain_community.llms import VertexAI # deprecated in langchain-community 0.0.12 and will be removed in 0.2.0.

st.set_page_config(page_title="LangChain Agent", layout="wide")


def main():
    # Streamlit UI elements
    st.title("LangGraph Agent + Gemini Pro + Custom Tool + Streamlit")

    # Input from user
    input_text = st.text_area("Enter your text:")

    if st.button("Run Agent"):
        # os.environ["SERPER_API_KEY"] = "YOUR-KEY-API"
        # search = GoogleSerperAPIWrapper()

        # Configure OpenAI settings
        # os.environ["OPENAI_API_KEY"] = "YOUR KEY"
        # lmstudio_llm = OpenAI(temperature=0, openai_api_base="http://localhost:1234/v1")
        search = DuckDuckGoSearchRun()

        def toggle_case(word):
            toggled_word = ""
            for char in word:
                if char.islower():
                    toggled_word += char.upper()
                elif char.isupper():
                    toggled_word += char.lower()
                else:
                    toggled_word += char
            return toggled_word

        def sort_string(string):
            return ''.join(sorted(string))

        tools = [
            Tool(
                name="Search",
                func=search.run,
                description="useful for when you need to answer questions about current events",
            ),
            Tool(
                name="Toogle_Case",
                func=lambda word: toggle_case(word),
                description="use when you want covert the letter to uppercase or lowercase",
            ),
            Tool(
                name="Sort String",
                func=lambda string: sort_string(string),
                description="use when you want sort a string alphabetically",
            ),

        ]

        prompt = hub.pull("hwchase17/react")

        # llm = ChatGoogleGenerativeAI(model="gemini-pro",
        #                              google_api_key="Your_API_KEY",
        #                              convert_system_message_to_human=True,
        #                              verbose=True,
        #                              )
        # llm = ChatVertexAI(convert_system_message_to_human=True, model="gemini-pro", verbose=True)
        llm = ChatVertexAI()  # VertexAI()
        agent_runnable = create_react_agent(llm, tools, prompt)

        # To maintain internal state, to be used in LangGraph
        class AgentState(TypedDict):
            input: str
            chat_history: list[BaseMessage]
            agent_outcome: Union[AgentAction, AgentFinish, None]
            return_direct: bool
            intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]

        tool_executor = ToolExecutor(tools)

        def run_agent(state):
            """
            #if you want to better manages intermediate steps
            inputs = state.copy()
            if len(inputs['intermediate_steps']) > 5:
                inputs['intermediate_steps'] = inputs['intermediate_steps'][-5:]
            """
            agent_outcome = agent_runnable.invoke(state)
            return {"agent_outcome": agent_outcome}

        def execute_tools(state):
            messages = [state['agent_outcome']]
            last_message = messages[-1]
            ######### human in the loop ###########
            # human input y/n
            # Get the most recent agent_outcome - this is the key added in the `agent` above
            # state_action = state['agent_outcome']
            # human_key = input(f"[y/n] continue with: {state_action}?")
            # if human_key == "n":
            #     raise ValueError

            tool_name = last_message.tool
            arguments = last_message
            if tool_name == "Search" or tool_name == "Sort" or tool_name == "Toggle_Case":

                if "return_direct" in arguments:
                    del arguments["return_direct"]
            action = ToolInvocation(
                tool=tool_name,
                tool_input=last_message.tool_input,
            )
            response = tool_executor.invoke(action)
            return {"intermediate_steps": [(state['agent_outcome'], response)]}

        def should_continue(state):
            messages = [state['agent_outcome']]
            last_message = messages[-1]
            if "Action" not in last_message.log:
                return "end"
            else:
                arguments = state["return_direct"]
                if arguments is True:
                    return "final"
                else:
                    return "continue"

        def first_agent(inputs):
            action = AgentActionMessageLog(
                tool="Search",
                tool_input=inputs["input"],
                log="",
                message_log=[]
            )
            return {"agent_outcome": action}

        # Define Langgraph
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", run_agent)
        workflow.add_node("action", execute_tools)
        workflow.add_node("final", execute_tools)
        # uncomment if you want to always calls a certain tool first
        # workflow.add_node("first_agent", first_agent) work

        workflow.set_entry_point("agent")
        # uncomment if you want to always calls a certain tool first
        # workflow.set_entry_point("first_agent")

        workflow.add_conditional_edges("agent",
                                       should_continue,
                                       {
                                           "continue": "action",
                                           "final": "final",
                                           "end": END
                                       })

        workflow.add_edge('action', 'agent')
        workflow.add_edge('final', END)
        # uncomment if you want to always calls a certain tool first
        # workflow.add_edge('first_agent', 'action')
        app = workflow.compile()  # it has become langchain runnable object

        inputs = {"input": input_text, "chat_history": [], "return_direct": False}
        results = []
        for s in app.stream(inputs, {"recursion_limit": 150}):
            result = list(s.values())[0]
            results.append(result)
            st.write(result)  # Display each step's output

        # result = app.invoke({"input": input_text, "chat_history": [], "return_direct": False})
        #
        # print(result["agent_outcome"].return_values["output"])


if __name__ == "__main__":
    main()
