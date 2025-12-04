from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage


# Initialize Groq LLM (ensure GROQ_API_KEY is set in your environment)
model = ChatGroq(model="llama-3.3-70b-versatile")  # https://console.groq.com/docs/models

# # or assuming os.environ["ANTHROPIC_API_KEY"] = "sk-..."
# model = init_chat_model(
#     "claude-sonnet-4-5-20250929",
#     # Kwargs passed to the model:
#     temperature=0.7,
#     timeout=30,
#     max_tokens=1000,
# )

conversation = [
    SystemMessage("You are a helpful assistant that translates English to French."),
    HumanMessage("Translate: I love programming."),
    AIMessage("J'adore la programmation."),
    HumanMessage("Translate: I love building applications.")
]

response = model.invoke(conversation)
print(response)  # AIMessage("J'adore créer des applications.")