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
        'model': 'Mistral 7B Instruct v01 Q2',  # 'llama 7B q4_0 ggml'
        'api_key': 'any string here is fine',
        'api_type': 'openai',
        'api_base': "http://localhost:1234/v1",
        'api_version': '2023-05-15'
    }
]
llm_config = {
    "functions": [
        {
            "name": "python",
            "description": "run cell in ipython and return the execution result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cell": {
                        "type": "string",
                        "description": "Valid Python cell to execute.",
                    }
                },
                "required": ["cell"],
            },
        },
        {
            "name": "sh",
            "description": "run a shell script and return the execution result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Valid shell script to execute.",
                    }
                },
                "required": ["script"],
            },
        },
    ],
    "config_list": local_config_list,
    "request_timeout": 120,
}
chatbot = autogen.AssistantAgent(
    name="chatbot",
    system_message="For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the " +
                   "task is done.",
    llm_config=llm_config,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    code_execution_config={"work_dir": "coding"},
)

# define functions according to the function desription
from IPython import get_ipython


def exec_python(cell):
    ipython = get_ipython()
    result = ipython.run_cell(cell)
    log = str(result.result)
    if result.error_before_exec is not None:
        log += f"\n{result.error_before_exec}"
    if result.error_in_exec is not None:
        log += f"\n{result.error_in_exec}"
    return log


def exec_sh(script):
    return user_proxy.execute_code_blocks([("sh", script)])


# register the functions
user_proxy.register_function(
    function_map={
        "python": exec_python,
        "sh": exec_sh,
    }
)

# start the conversation
user_proxy.initiate_chat(
    chatbot,
    message="Draw two agents chatting with each other with an example dialog. Don't add plt.show().",
)
