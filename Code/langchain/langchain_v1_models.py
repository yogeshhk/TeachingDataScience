from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatLlamaCpp
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage


# Initialize Groq LLM (ensure GROQ_API_KEY is set in your environment)
# model = ChatGroq(model="llama-3.3-70b-versatile")  # https://console.groq.com/docs/models

# model = init_chat_model(
#     "groq:llama-3.3-70b-versatile",
#     # model_provider="groq",
#     temperature=0.7,
#     timeout=30,
#     max_tokens=1000,
# )

# Local Qwen3-1.7B via llama.cpp, in-process, no server (see
# TODO_code_consolidation_and_local_llm.md Part 2). CPU-only, ~17-24 tok/s on this
# machine. Append "/no_think" to the final human message: Qwen3 defaults to a
# "thinking" mode that otherwise consumes the whole max_tokens budget before
# producing an answer.
model = ChatLlamaCpp(
    model_path=r"D:\Yogesh\models\lmstudio-community\Qwen3-1.7B-GGUF\Qwen3-1.7B-Q4_K_M.gguf",
    n_ctx=4096,
    temperature=0.7,
    max_tokens=1000,
    n_gpu_layers=0,
    verbose=False,
)

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications. /no_think")
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")