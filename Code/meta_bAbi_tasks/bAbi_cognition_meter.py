# Reference paper:
# "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks" https://arxiv.org/abs/1502.05698
# Reference blog:
# https://medium.com/technology-hits/does-chatgpt-really-understand-the-language-4855683b0143

# Assuming OPENAI_API_KEY set in Environment variables

from langchain.llms import OpenAI, HuggingFaceHub
from langchain import PromptTemplate
from langchain import LLMChain
import pandas as pd
from colorama import Fore

bool_score = True
total_score = 0
count = 0

template = "{context} {query}"
prompt = PromptTemplate(template=template, input_variables=['context', 'query'])

llms = [{'name': 'Flan', 'model': HuggingFaceHub(repo_id="google/flan-t5-small", model_kwargs={"temperature": 1e-10})},
        {'name': 'OpenAI', 'model': OpenAI(temperature=0)}]

df = pd.read_excel(r'data/Test2.xlsx')

for llm_dict in llms:
    llm_name = llm_dict['name']
    llm_model = llm_dict['model']
    chain = LLMChain(llm=llm_model, prompt=prompt)

    df.reset_index()
    for index, row in df.iterrows():
        context = (row['Context']).replace("\n", " ")
        queries = (row['Queries']).split("\n")
        labels = (row['Labels']).split("\n")
        for query, label in zip(queries, labels):
            print(Fore.BLUE + f"Context: {context}")
            print(Fore.YELLOW + f"Query:{query}")
            print(Fore.GREEN + f"Label: {label}")
            keywords = {'context': context, 'query': query}
            print(Fore.MAGENTA + f"{llm_name} Response: {chain.run(keywords).strip()}")
            if bool_score:
                str_score = input(Fore.RED + 'Score? 0 for Wrong, 1 for Perfect : ')
                total_score += float(str_score)
                count += 1
            print(Fore.CYAN + "---")

    if count:
        print(f"Overall score for {llm_name}: {(total_score / count):.2f}")
    print(Fore.CYAN + "===========================")
