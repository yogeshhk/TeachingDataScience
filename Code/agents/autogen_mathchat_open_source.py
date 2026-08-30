# https://github.com/microsoft/autogen/blob/osllm/notebook/open_source_language_model_example.ipynb

# Following ways failed to start local llm server
# >> modelz-llm -m bigscience/bloomz-560m --device auto [NOT FOR WINDOWS]
# >> python -m llama_cpp.server --model <model path>.gguf

# Worked with LMStudio. You can download models from UI or if you have them already, keep them in
# C:\Users\yoges\.cache\lm-studio\models\yogeshhk\Sarvadnya , 'llama-7b.ggmlv3.q4_0.bin' was recognized
# Check using CHAT if it responds well.
# Start server, take the base_path URL and set it as below, at both places.
# Then run this file

# Setup autogen with the correct API
import autogen
from autogen import AssistantAgent, UserProxyAgent
from autogen.agentchat.contrib.math_user_proxy_agent import MathUserProxyAgent

import openai

openai.api_type = "openai"
openai.api_key = "..."
openai.api_base = "http://localhost:1234/v1"
openai.api_version = "2023-05-15"

autogen.oai.ChatCompletion.start_logging()

local_config_list = [
    {
        'model': 'llama 7B q4_0 ggml',
        'api_key': 'any string here is fine',
        'api_type': 'openai',
        'api_base': "http://localhost:1234/v1",
        'api_version': '2023-05-15'
    }
]
# 1. create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config={
        "request_timeout": 600,
        "seed": 42,
        "config_list": local_config_list,
    }
)

# 2. create the MathUserProxyAgent instance named "mathproxyagent"
# By default, the human_input_mode is "NEVER", which means the agent will not ask for human input.
mathproxyagent = MathUserProxyAgent(
    name="mathproxyagent",
    human_input_mode="NEVER",
    code_execution_config={"use_docker": False},
)

# given a math problem, we use the mathproxyagent to generate a prompt to be sent to the assistant as the initial
# message. the assistant receives the message and generates a response. The response will be sent back to the
# mathproxyagent for processing. The conversation continues until the termination condition is met, in MathChat,
# the termination condition is the detect of "\boxed{}" in the response.
math_problem = "Find all $x$ that satisfy the inequality $(2x+10)(x+3)<(3x+9)(x+8)$. Express your answer in interval " \
               "notation."
mathproxyagent.initiate_chat(assistant, problem=math_problem)
