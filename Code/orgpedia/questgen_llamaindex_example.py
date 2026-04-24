# https://gpt-index.readthedocs.io/en/latest/examples/evaluation/QuestionGeneration.html

from llama_index import KeywordTableIndex, SimpleDirectoryReader, LLMPredictor, ServiceContext
from llama_index.evaluation import DatasetGenerator
from langchain import HuggingFaceHub

documents = SimpleDirectoryReader('data/experiment').load_data()

repo_id = "tiiuae/falcon-7b"

llm_predictor = LLMPredictor(llm=HuggingFaceHub(repo_id=repo_id,
                                                model_kwargs={"temperature": 0.1, 'truncation': 'only_first',
                                                              "max_length": 512}))
service_context = ServiceContext.from_defaults(chunk_size=64, llm_predictor=llm_predictor)

data_generator = DatasetGenerator.from_documents(documents, service_context=service_context)
eval_questions = data_generator.generate_questions_from_nodes()
print(eval_questions)
