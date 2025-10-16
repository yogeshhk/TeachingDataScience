# Frequently Used Procedures

## Steps to test new model on Ludwig fine-tuning 

https://medium.com/google-cloud/gemma-for-gst-4595d5f60b6b
Need HuggingFace API Token, access approval to Gemma–7b-it,

```
dataset:
url = "https://raw.githubusercontent.com/yogeshhk/Sarvadnya/master/src/ludwig/data/cbic-gst_gov_in_fgaq.csv"

# Download the file
wget.download(url, 'cbic-gst_gov_in_fgaq.csv')
df = pd.read_csv('cbic-gst_gov_in_fgaq.csv', encoding='cp1252')
```
Training:
```
instruction_tuning_yaml = yaml.safe_load("""
model_type: llm
base_model: google/gemma-7b-it

quantization:
 bits: 4

adapter:
 type: lora

prompt:
  template: |
    ### Instruction:
    You are a taxation expert on Goods and Services Tax used in India.
    Take the Input given below which is a Question. Give Answer for it as a Response.

    ### Input:
    {Question}

    ### Response:

input_features:
 - name: Question
   type: text
   preprocessing:
      max_sequence_length: 1024

output_features:
 - name: Answer
   type: text
   preprocessing:
      max_sequence_length: 384

trainer:
  type: finetune
  epochs: 8
  batch_size: 1
  eval_batch_size: 2
  gradient_accumulation_steps: 16  # effective batch size = batch size * gradient_accumulation_steps
  learning_rate: 2.0e-4
  enable_gradient_checkpointing: true
  learning_rate_scheduler:
    decay: cosine
    warmup_fraction: 0.03
    reduce_on_plateau: 0

generation:
  temperature: 0.1
  max_new_tokens: 512

backend:
 type: local
""")

model_instruction_tuning = LudwigModel(config=instruction_tuning_yaml,  logging_level=logging.INFO)
results_instruction_tuning = model_instruction_tuning.train(dataset=df)
```

Prediction:
```
test_df = pd.DataFrame([
    {
        "Question": "If I am not an existing taxpayer and wish to newly register under GST, when can I do so?"
    },
    {
        "Question": "Does aggregate turnover include value of inward supplies received on which RCM is payable?"
    },
])


predictions_instruction_tuning_df, output_directory = model_instruction_tuning.predict(dataset=test_df)
print(predictions_instruction_tuning_df["Answer_response"].tolist())
```

## Steps to use LM Studio instead of OpenAI APIs

- You can download models from LM Studio UI or if you have them already, keep them in “C:\Users\<windows login>\.cache\lm-studio\models\<author>\<repo>”
- Using ‘openhermes-2.5-mistral-7b.Q4_0.gguf’ here
- Check using CHAT if it responds well.
- Start server, take the openai_api_baseURL and set it as below.
```
from langchain_openai.llms import OpenAI

# Configure OpenAI settings
# os.environ["OPENAI_API_KEY"] = "YOUR KEY"

--

import openai

# Configure OpenAI settings
openai.api_type = "openai"
openai.api_key = "..."
openai.api_base = "http://localhost:1234/v1"
openai.api_version = "2023-05-15"

--
lmstudio_llm = OpenAI(temperature=0, openai_api_base="http://localhost:1234/v1")

llm=lmstudio_llm

```

## Steps to test new model on GST RAG  Langchain

https://medium.com/google-cloud/building-a-gst-faqs-app-d8d903eb9c6

```
    loader = CSVLoader(file_path='./data/nlp_faq_engine_faqs.csv')
    docs = loader.load()

    # loader = PyPDFLoader("./data/Final-GST-FAQ-edition.pdf")
    # docs = loader.load_and_split()

    loader = UnstructuredHTMLLoader("data/cbic-gst_gov_in_fgaq.html")
    docs += loader.load()

    embeddings = HuggingFaceHubEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriver = db.as_retriever()
    llm = VertexAI() # or any other
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriver, verbose=False, chain_type="stuff")
    return chain

response = chain.run(question)
```

## Steps to use GCP LLM

- Create a new project or select an existing one in the GCP Console, after login.
- Click on Dashboard to access the project dashboard and select “APIs & Services” from the burger menu. If Vertex AI API is not listed under Enabled APIs & Services, use the search bar at the top of the page to search for Vertex AI API, click on the search result, and enable it.
- To interact with Vertex AI through the API, you need to set up authentication. This typically involves creating a service account and downloading the corresponding key (typically a JSON file). In the GCP Console, navigate to IAM & Admin > Service Accounts
- Create a new service account. 
- Download a JSON-formatted private key. This key will be used for authentication between your application and the Vertex AI API.
- Set ENV variable GOOGLE_APPLICATION_CREDENTIALS="<path to the JSON key file obtained in the authentication setup step>"
- Mine is at C:\Users\yoges\Documents\document-ai-374204-85985df1d53e.json
- Use it in Langchain like
from langchain.llms import VertexAI 
llm = VertexAI() # model_name="gemini-pro", deafult=

https://ai.gopubby.com/get-started-with-google-vertex-ai-api-and-langchain-integration-360262d05216

https://python.langchain.com/docs/integrations/llms/google_vertex_ai_palm/