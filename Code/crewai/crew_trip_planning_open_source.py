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
city_selection_agent = Agent(
    role='Tourist City Selection Expert',
    goal='Select the best islands based on the destinations where Indians can go',
    backstory="""An expert in picking ideal destinations and you are aware of latest geopolitical situations""",
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm=lmstudio_llm
)
travel_planning_agent = Agent(
    role='Amazing Travel Package Designer',
    goal='Create the most amazing travel itineraries with budget and packing suggestions for the destination',
    backstory="""Specialist in travel planning and logistics with decades of experience""",
    verbose=True,
    allow_delegation=False,
    llm=lmstudio_llm
)

# Create tasks for your agents
task1 = Task(
    description="""Identify favorable tourist islands having beaches, which are friendly to Indians. 
    They can be domestic ie within Indian territory or in the Indian ocean""",
    agent=city_selection_agent
)

task2 = Task(
    description="""Expand this guide into a a full 7-day travel 
        itinerary for the selected destination.
        Have detailed per-day plans, including 
        weather forecasts, places to eat, packing suggestions, 
        and a budget breakdown.
        
        You MUST suggest actual places to visit, actual hotels 
        to stay and actual restaurants to go to.
        
        This itinerary should cover all aspects of the trip, 
        from arrival to departure, integrating the city guide
        information with practical travel logistics.
        
        Your final answer MUST be a complete expanded travel plan,
        formatted as markdown, encompassing a daily schedule,
        anticipated weather conditions, recommended clothing and
        items to pack, and a detailed budget, ensuring THE BEST
        TRIP EVER, Be specific and give it a reason why you picked
        # up each place, what make them special!""",
    agent=travel_planning_agent
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[city_selection_agent, travel_planning_agent],
    tasks=[task1, task2],
    verbose=2,  # You can set it to 1 or 2 to different logging levels
    process=Process.sequential
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)
