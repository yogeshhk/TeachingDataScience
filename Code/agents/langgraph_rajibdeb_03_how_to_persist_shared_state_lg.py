# Author: Rajib Deb
# A simple example showing how langgraph persists state
import operator
import os
import pickle
from typing import TypedDict, Annotated, Sequence

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig, ConfigurableFieldSpec
from langchain_openai import ChatOpenAI
from langgraph.checkpoint import BaseCheckpointSaver, Checkpoint
from langgraph.graph import MessageGraph

from chkpt_client.sqllite_client import SQLLiteClient

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class DBCheckPointer(BaseCheckpointSaver):
    class Config:
        arbitrary_types_allowed = True

    client = SQLLiteClient()

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id="session_id",
                annotation=str,
                name="Session ID",
                description=None,
                default="",
                is_shared=True,
            ),
        ]

    def get(self, config: RunnableConfig) -> Checkpoint:
        checkpoint = self.client.select_chkpt(config["configurable"]["session_id"])
        return checkpoint

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        record = (config["configurable"]["session_id"],
                  pickle.dumps(checkpoint),)
        try:
            self.client.insert_chkpt(record=record)
        except Exception as e:
            print(e)


model = ChatOpenAI(temperature=0, streaming=False)


def personal_assistant(messages):
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return response


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


workflow = MessageGraph()
workflow.add_node("assistant", personal_assistant)

workflow.set_entry_point("assistant")
workflow.set_finish_point("assistant")

checkpoint = DBCheckPointer()
app = workflow.compile(checkpointer=checkpoint)

while True:
    content = input("Ask me a question \n")
    if content == "exit":
        exit(0)
    human_message = [HumanMessage(content=content)]
    response = app.invoke(human_message, {"configurable": {"session_id": "2"}})
    print(response[-1].content)
