# Original https://github.com/joaomdmoura/crewAI-examples/tree/main/trip_planner

import os
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from langchain.tools import DuckDuckGoSearchRun
import openai
# from langchain.llms import OpenAI
from langchain_openai.llms import OpenAI
from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools

# Configure OpenAI settings
# os.environ["OPENAI_API_KEY"] = "YOUR KEY"

lmstudio_llm = OpenAI(temperature=0, openai_api_base="http://localhost:1234/v1")

search_tool = DuckDuckGoSearchRun()

class JobAgencyAgents():

  def city_selection_agent(self):
    return Agent(
        role='City Selection Expert',
        goal='Select the best city based on weather, season, and prices',
        backstory=
        'An expert in analyzing travel data to pick ideal destinations',
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        verbose=True)

  def local_expert(self):
    return Agent(
        role='Local Expert at this city',
        goal='Provide the BEST insights about the selected city',
        backstory="""A knowledgeable local guide with extensive information
        about the city, it's attractions and customs""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
        ],
        verbose=True)

  def travel_concierge(self):
    return Agent(
        role='Amazing Travel Concierge',
        goal="""Create the most amazing travel itineraries with budget and 
        packing suggestions for the city""",
        backstory="""Specialist in travel planning and logistics with 
        decades of experience""",
        tools=[
            SearchTools.search_internet,
            BrowserTools.scrape_and_summarize_website,
            CalculatorTools.calculate,
        ],
        verbose=True)

class JobAgencyCrew:

    def __init__(self, education, experience, job_descriptions, skills):
        self.experience = experience
        self.education = education
        self.skills = skills
        self.job_descriptions = job_descriptions

    def run(self):
        agents = JobAgencyAgents()
        tasks = TripTasks()

        city_selector_agent = agents.city_selection_agent()
        local_expert_agent = agents.local_expert()
        travel_concierge_agent = agents.travel_concierge()

        identify_task = tasks.identify_task(
            city_selector_agent,
            self.education,
            self.experience,
            self.skills,
            self.job_descriptions
        )
        gather_task = tasks.gather_task(
            local_expert_agent,
            self.education,
            self.skills,
            self.job_descriptions
        )
        plan_task = tasks.plan_task(
            travel_concierge_agent,
            self.education,
            self.skills,
            self.job_descriptions
        )

        crew = Crew(
            agents=[
                city_selector_agent, local_expert_agent, travel_concierge_agent
            ],
            tasks=[identify_task, gather_task, plan_task],
            verbose=True
        )

        result = crew.kickoff()
        return result


if __name__ == "__main__":
    print("## Welcome to Trip Planner Crew")
    print('-------------------------------')
    location = input(
        dedent("""
      From where will you be traveling from?
    """))
    cities = input(
        dedent("""
      What are the cities options you are interested in visiting?
    """))
    date_range = input(
        dedent("""
      What is the date range you are interested in traveling?
    """))
    interests = input(
        dedent("""
      What are some of your high level interests and hobbies?
    """))

    trip_crew = JobAgencyCrew(location, cities, date_range, interests)
    result = trip_crew.run()
    print("\n\n########################")
    print("## Here is you Trip Plan")
    print("########################\n")
    print(result)