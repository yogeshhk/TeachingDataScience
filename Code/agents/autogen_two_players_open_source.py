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

# # Perform Completion
# question = "Who are you? Tell it in 2 lines only."
# response = autogen.oai.Completion.create(config_list=local_config_list, prompt=question, temperature=0)
# ans = autogen.oai.Completion.extract_text(response)[0]
#
# print("Answer is:", ans)
#
# # Student Teacher
#
# small = AssistantAgent(name="small model",
#                        max_consecutive_auto_reply=2,
#                        system_message="You should act as a student! Give response in 2 lines only.",
#                        llm_config={
#                            "config_list": local_config_list,
#                            "temperature": 0.5,
#                        })
#
# big = AssistantAgent(name="big model",
#                      max_consecutive_auto_reply=2,
#                      system_message="Act as a teacher.Give response in 2 lines only.",
#                      llm_config={
#                          "config_list": local_config_list,
#                          "temperature": 0.5,
#                      })
#
# big.initiate_chat(small, message="Who are you?")

# Entrepreneur - Accountant

ennreprenuer = AssistantAgent(name="Entrepreneur",
                              max_consecutive_auto_reply=2,
                              system_message="Act as a Entrepreneur! You want to get the task done from the Accountant",
                              llm_config={
                                  "config_list": local_config_list,
                                  "temperature": 0.5,
                              })

accountant = AssistantAgent(name="Accountant",
                            max_consecutive_auto_reply=2,
                            system_message="Act as a Accountant. You want to help the Entrepreneur to get the task done",
                            llm_config={
                                "config_list": local_config_list,
                                "temperature": 0.5,
                            })

accountant.initiate_chat(ennreprenuer, message="I want to help prepare and file the taxes.")
