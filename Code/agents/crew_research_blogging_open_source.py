# Ref video CrewAI Tutorial - Next Generation AI Agent Teams (Fully Local) Matthew Berman
# https://www.youtube.com/watch?v=tnejrr-0a94
# https://github.com/joaomdmoura/crewAI
# Using Local LLMs LM Studio Way ############
# https://medium.com/analytics-vidhya/microsoft-autogen-using-open-source-models-97cba96b0f75

import os
from crewai import Agent, Task, Crew, Process
from langchain.tools import DuckDuckGoSearchRun
import openai
# from langchain.llms import OpenAI
from langchain_openai.llms import OpenAI

# Configure OpenAI settings
# os.environ["OPENAI_API_KEY"] = "YOUR KEY"
lmstudio_llm = OpenAI(temperature=0, openai_api_base="http://localhost:1234/v1")

search_tool = DuckDuckGoSearchRun()

# Define your agents with roles and goals
coach = Agent(
    role='Senior Career Coach',
    goal="""Find and explore the most exciting career skills related to tech and AI in 2024""",
    backstory="""You are and Expert Career Coach that knows how to spot
              emerging trends and skills needed in AI and tech. 
              """,
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=lmstudio_llm
)
influencer = Agent(
    role='LinkedIn Influencer Writer',
    goal="""Write engaging and interesting LinkedIn post in maximum 200 words. Add relevant emojis""",
    backstory="""You are an Expert Writer on LinkedIn in the field of AI and tech""",
    verbose=True,
    allow_delegation=True,
    llm=lmstudio_llm
)
critic = Agent(
    role='Expert Writing Critic',
    goal="""Provide feedback and criticize post drafts. """,
    backstory="""You are an Expert at providing feedback to the technical writers. 
    Make sure that the the suggestions are actionable, compelling, simple and concise.
    Also make sure that the post is within 200 words, has emojis and relevant hashtags.""",
    verbose=True,
    allow_delegation=True,
    llm=lmstudio_llm
)

# Create tasks for your agents
task_search = Task(
    description="""Make a detailed report on the latest rising skills 
                  in AI and tech space. Your final answer MUST be a list of at least 5 exciting new AI and tech skills
                in the format of bullet points.""",
    agent=coach
)

task_post = Task(
    description="""Write a LinkedIn post with a short but impactful headline 
                  and at max 200 words. It should list the latest 
                  AI and tech skills which are going to be in demand. 
                  """,
    agent=influencer
)

task_critique = Task(
    description="""Identify parts of the post that aren't written concise enough
                   and improve them. Make sure that the post has engaging 
                  headline with 30 characters max, and that there are at max 200 words. 
                  """,
    agent=critic
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[coach, influencer, critic],
    tasks=[task_search, task_post, task_critique],
    verbose=2,
    process=Process.sequential  # Sequential process will have tasks executed one after the other
    # and the outcome of the previous one is passed as extra content into this next.
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
